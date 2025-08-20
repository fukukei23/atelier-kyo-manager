
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\f2py2e.py ===
"""

f2py2e - Fortran to Python C/API generator. 2nd Edition.
         See __usage__ below.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
import argparse
import os
import pprint
import re
import sys

from numpy.f2py._backends import f2py_build_generator

from . import (
    __version__,
    auxfuncs,
    capi_maps,
    cb_rules,
    cfuncs,
    crackfortran,
    f90mod_rules,
    rules,
)
from .cfuncs import errmess

f2py_version = __version__.version
numpy_version = __version__.version

# outmess=sys.stdout.write
show = pprint.pprint
outmess = auxfuncs.outmess
MESON_ONLY_VER = (sys.version_info >= (3, 12))

__usage__ =\
f"""Usage:

1) To construct extension module sources:

      f2py [<options>] <fortran files> [[[only:]||[skip:]] \\
                                        <fortran functions> ] \\
                                       [: <fortran files> ...]

2) To compile fortran files and build extension modules:

      f2py -c [<options>, <build_flib options>, <extra options>] <fortran files>

3) To generate signature files:

      f2py -h <filename.pyf> ...< same options as in (1) >

Description: This program generates a Python C/API file (<modulename>module.c)
             that contains wrappers for given fortran functions so that they
             can be called from Python. With the -c option the corresponding
             extension modules are built.

Options:

  -h <filename>    Write signatures of the fortran routines to file <filename>
                   and exit. You can then edit <filename> and use it instead
                   of <fortran files>. If <filename>==stdout then the
                   signatures are printed to stdout.
  <fortran functions>  Names of fortran routines for which Python C/API
                   functions will be generated. Default is all that are found
                   in <fortran files>.
  <fortran files>  Paths to fortran/signature files that will be scanned for
                   <fortran functions> in order to determine their signatures.
  skip:            Ignore fortran functions that follow until `:'.
  only:            Use only fortran functions that follow until `:'.
  :                Get back to <fortran files> mode.

  -m <modulename>  Name of the module; f2py generates a Python/C API
                   file <modulename>module.c or extension module <modulename>.
                   Default is 'untitled'.

  '-include<header>'  Writes additional headers in the C wrapper, can be passed
                      multiple times, generates #include <header> each time.

  --[no-]lower     Do [not] lower the cases in <fortran files>. By default,
                   --lower is assumed with -h key, and --no-lower without -h key.

  --build-dir <dirname>  All f2py generated files are created in <dirname>.
                   Default is tempfile.mkdtemp().

  --overwrite-signature  Overwrite existing signature file.

  --[no-]latex-doc Create (or not) <modulename>module.tex.
                   Default is --no-latex-doc.
  --short-latex    Create 'incomplete' LaTeX document (without commands
                   \\documentclass, \\tableofcontents, and \\begin{{document}},
                   \\end{{document}}).

  --[no-]rest-doc Create (or not) <modulename>module.rst.
                   Default is --no-rest-doc.

  --debug-capi     Create C/API code that reports the state of the wrappers
                   during runtime. Useful for debugging.

  --[no-]wrap-functions    Create Fortran subroutine wrappers to Fortran 77
                   functions. --wrap-functions is default because it ensures
                   maximum portability/compiler independence.

  --[no-]freethreading-compatible    Create a module that declares it does or
                   doesn't require the GIL. The default is
                   --freethreading-compatible for backward
                   compatibility. Inspect the Fortran code you are wrapping for
                   thread safety issues before passing
                   --no-freethreading-compatible, as f2py does not analyze
                   fortran code for thread safety issues.

  --include-paths <path1>:<path2>:...   Search include files from the given
                   directories.

  --help-link [..] List system resources found by system_info.py. See also
                   --link-<resource> switch below. [..] is optional list
                   of resources names. E.g. try 'f2py --help-link lapack_opt'.

  --f2cmap <filename>  Load Fortran-to-Python KIND specification from the given
                   file. Default: .f2py_f2cmap in current directory.

  --quiet          Run quietly.
  --verbose        Run with extra verbosity.
  --skip-empty-wrappers   Only generate wrapper files when needed.
  -v               Print f2py version ID and exit.


build backend options (only effective with -c)
[NO_MESON] is used to indicate an option not meant to be used
with the meson backend or above Python 3.12:

  --fcompiler=         Specify Fortran compiler type by vendor [NO_MESON]
  --compiler=          Specify distutils C compiler type [NO_MESON]

  --help-fcompiler     List available Fortran compilers and exit [NO_MESON]
  --f77exec=           Specify the path to F77 compiler [NO_MESON]
  --f90exec=           Specify the path to F90 compiler [NO_MESON]
  --f77flags=          Specify F77 compiler flags
  --f90flags=          Specify F90 compiler flags
  --opt=               Specify optimization flags [NO_MESON]
  --arch=              Specify architecture specific optimization flags [NO_MESON]
  --noopt              Compile without optimization [NO_MESON]
  --noarch             Compile without arch-dependent optimization [NO_MESON]
  --debug              Compile with debugging information

  --dep                <dependency>
                       Specify a meson dependency for the module. This may
                       be passed multiple times for multiple dependencies.
                       Dependencies are stored in a list for further processing.

                       Example: --dep lapack --dep scalapack
                       This will identify "lapack" and "scalapack" as dependencies
                       and remove them from argv, leaving a dependencies list
                       containing ["lapack", "scalapack"].

  --backend            <backend_type>
                       Specify the build backend for the compilation process.
                       The supported backends are 'meson' and 'distutils'.
                       If not specified, defaults to 'distutils'. On
                       Python 3.12 or higher, the default is 'meson'.

Extra options (only effective with -c):

  --link-<resource>    Link extension module with <resource> as defined
                       by numpy.distutils/system_info.py. E.g. to link
                       with optimized LAPACK libraries (vecLib on MacOSX,
                       ATLAS elsewhere), use --link-lapack_opt.
                       See also --help-link switch. [NO_MESON]

  -L/path/to/lib/ -l<libname>
  -D<define> -U<name>
  -I/path/to/include/
  <filename>.o <filename>.so <filename>.a

  Using the following macros may be required with non-gcc Fortran
  compilers:
    -DPREPEND_FORTRAN -DNO_APPEND_FORTRAN -DUPPERCASE_FORTRAN

  When using -DF2PY_REPORT_ATEXIT, a performance report of F2PY
  interface is printed out at exit (platforms: Linux).

  When using -DF2PY_REPORT_ON_ARRAY_COPY=<int>, a message is
  sent to stderr whenever F2PY interface makes a copy of an
  array. Integer <int> sets the threshold for array sizes when
  a message should be shown.

Version:     {f2py_version}
numpy Version: {numpy_version}
License:     NumPy license (see LICENSE.txt in the NumPy source code)
Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
https://numpy.org/doc/stable/f2py/index.html\n"""


def scaninputline(inputline):
    files, skipfuncs, onlyfuncs, debug = [], [], [], []
    f, f2, f3, f5, f6, f8, f9, f10 = 1, 0, 0, 0, 0, 0, 0, 0
    verbose = 1
    emptygen = True
    dolc = -1
    dolatexdoc = 0
    dorestdoc = 0
    wrapfuncs = 1
    buildpath = '.'
    include_paths, freethreading_compatible, inputline = get_newer_options(inputline)
    signsfile, modulename = None, None
    options = {'buildpath': buildpath,
               'coutput': None,
               'f2py_wrapper_output': None}
    for l in inputline:
        if l == '':
            pass
        elif l == 'only:':
            f = 0
        elif l == 'skip:':
            f = -1
        elif l == ':':
            f = 1
        elif l[:8] == '--debug-':
            debug.append(l[8:])
        elif l == '--lower':
            dolc = 1
        elif l == '--build-dir':
            f6 = 1
        elif l == '--no-lower':
            dolc = 0
        elif l == '--quiet':
            verbose = 0
        elif l == '--verbose':
            verbose += 1
        elif l == '--latex-doc':
            dolatexdoc = 1
        elif l == '--no-latex-doc':
            dolatexdoc = 0
        elif l == '--rest-doc':
            dorestdoc = 1
        elif l == '--no-rest-doc':
            dorestdoc = 0
        elif l == '--wrap-functions':
            wrapfuncs = 1
        elif l == '--no-wrap-functions':
            wrapfuncs = 0
        elif l == '--short-latex':
            options['shortlatex'] = 1
        elif l == '--coutput':
            f8 = 1
        elif l == '--f2py-wrapper-output':
            f9 = 1
        elif l == '--f2cmap':
            f10 = 1
        elif l == '--overwrite-signature':
            options['h-overwrite'] = 1
        elif l == '-h':
            f2 = 1
        elif l == '-m':
            f3 = 1
        elif l[:2] == '-v':
            print(f2py_version)
            sys.exit()
        elif l == '--show-compilers':
            f5 = 1
        elif l[:8] == '-include':
            cfuncs.outneeds['userincludes'].append(l[9:-1])
            cfuncs.userincludes[l[9:-1]] = '#include ' + l[8:]
        elif l == '--skip-empty-wrappers':
            emptygen = False
        elif l[0] == '-':
            errmess(f'Unknown option {repr(l)}\n')
            sys.exit()
        elif f2:
            f2 = 0
            signsfile = l
        elif f3:
            f3 = 0
            modulename = l
        elif f6:
            f6 = 0
            buildpath = l
        elif f8:
            f8 = 0
            options["coutput"] = l
        elif f9:
            f9 = 0
            options["f2py_wrapper_output"] = l
        elif f10:
            f10 = 0
            options["f2cmap_file"] = l
        elif f == 1:
            try:
                with open(l):
                    pass
                files.append(l)
            except OSError as detail:
                errmess(f'OSError: {detail!s}. Skipping file "{l!s}".\n')
        elif f == -1:
            skipfuncs.append(l)
        elif f == 0:
            onlyfuncs.append(l)
    if not f5 and not files and not modulename:
        print(__usage__)
        sys.exit()
    if not os.path.isdir(buildpath):
        if not verbose:
            outmess(f'Creating build directory {buildpath}\n')
        os.mkdir(buildpath)
    if signsfile:
        signsfile = os.path.join(buildpath, signsfile)
    if signsfile and os.path.isfile(signsfile) and 'h-overwrite' not in options:
        errmess(
            f'Signature file "{signsfile}" exists!!! Use --overwrite-signature to overwrite.\n')
        sys.exit()

    options['emptygen'] = emptygen
    options['debug'] = debug
    options['verbose'] = verbose
    if dolc == -1 and not signsfile:
        options['do-lower'] = 0
    else:
        options['do-lower'] = dolc
    if modulename:
        options['module'] = modulename
    if signsfile:
        options['signsfile'] = signsfile
    if onlyfuncs:
        options['onlyfuncs'] = onlyfuncs
    if skipfuncs:
        options['skipfuncs'] = skipfuncs
    options['dolatexdoc'] = dolatexdoc
    options['dorestdoc'] = dorestdoc
    options['wrapfuncs'] = wrapfuncs
    options['buildpath'] = buildpath
    options['include_paths'] = include_paths
    options['requires_gil'] = not freethreading_compatible
    options.setdefault('f2cmap_file', None)
    return files, options


def callcrackfortran(files, options):
    rules.options = options
    crackfortran.debug = options['debug']
    crackfortran.verbose = options['verbose']
    if 'module' in options:
        crackfortran.f77modulename = options['module']
    if 'skipfuncs' in options:
        crackfortran.skipfuncs = options['skipfuncs']
    if 'onlyfuncs' in options:
        crackfortran.onlyfuncs = options['onlyfuncs']
    crackfortran.include_paths[:] = options['include_paths']
    crackfortran.dolowercase = options['do-lower']
    postlist = crackfortran.crackfortran(files)
    if 'signsfile' in options:
        outmess(f"Saving signatures to file \"{options['signsfile']}\"\n")
        pyf = crackfortran.crack2fortran(postlist)
        if options['signsfile'][-6:] == 'stdout':
            sys.stdout.write(pyf)
        else:
            with open(options['signsfile'], 'w') as f:
                f.write(pyf)
    if options["coutput"] is None:
        for mod in postlist:
            mod["coutput"] = f"{mod['name']}module.c"
    else:
        for mod in postlist:
            mod["coutput"] = options["coutput"]
    if options["f2py_wrapper_output"] is None:
        for mod in postlist:
            mod["f2py_wrapper_output"] = f"{mod['name']}-f2pywrappers.f"
    else:
        for mod in postlist:
            mod["f2py_wrapper_output"] = options["f2py_wrapper_output"]
    for mod in postlist:
        if options["requires_gil"]:
            mod['gil_used'] = 'Py_MOD_GIL_USED'
        else:
            mod['gil_used'] = 'Py_MOD_GIL_NOT_USED'
    return postlist


def buildmodules(lst):
    cfuncs.buildcfuncs()
    outmess('Building modules...\n')
    modules, mnames, isusedby = [], [], {}
    for item in lst:
        if '__user__' in item['name']:
            cb_rules.buildcallbacks(item)
        else:
            if 'use' in item:
                for u in item['use'].keys():
                    if u not in isusedby:
                        isusedby[u] = []
                    isusedby[u].append(item['name'])
            modules.append(item)
            mnames.append(item['name'])
    ret = {}
    for module, name in zip(modules, mnames):
        if name in isusedby:
            outmess('\tSkipping module "%s" which is used by %s.\n' % (
                name, ','.join('"%s"' % s for s in isusedby[name])))
        else:
            um = []
            if 'use' in module:
                for u in module['use'].keys():
                    if u in isusedby and u in mnames:
                        um.append(modules[mnames.index(u)])
                    else:
                        outmess(
                            f'\tModule "{name}" uses nonexisting "{u}" '
                            'which will be ignored.\n')
            ret[name] = {}
            dict_append(ret[name], rules.buildmodule(module, um))
    return ret


def dict_append(d_out, d_in):
    for (k, v) in d_in.items():
        if k not in d_out:
            d_out[k] = []
        if isinstance(v, list):
            d_out[k] = d_out[k] + v
        else:
            d_out[k].append(v)


def run_main(comline_list):
    """
    Equivalent to running::

        f2py <args>

    where ``<args>=string.join(<list>,' ')``, but in Python.  Unless
    ``-h`` is used, this function returns a dictionary containing
    information on generated modules and their dependencies on source
    files.

    You cannot build extension modules with this function, that is,
    using ``-c`` is not allowed. Use the ``compile`` command instead.

    Examples
    --------
    The command ``f2py -m scalar scalar.f`` can be executed from Python as
    follows.

    .. literalinclude:: ../../source/f2py/code/results/run_main_session.dat
        :language: python

    """
    crackfortran.reset_global_f2py_vars()
    f2pydir = os.path.dirname(os.path.abspath(cfuncs.__file__))
    fobjhsrc = os.path.join(f2pydir, 'src', 'fortranobject.h')
    fobjcsrc = os.path.join(f2pydir, 'src', 'fortranobject.c')
    # gh-22819 -- begin
    parser = make_f2py_compile_parser()
    args, comline_list = parser.parse_known_args(comline_list)
    pyf_files, _ = filter_files("", "[.]pyf([.]src|)", comline_list)
    # Checks that no existing modulename is defined in a pyf file
    # TODO: Remove all this when scaninputline is replaced
    if args.module_name:
        if "-h" in comline_list:
            modname = (
                args.module_name
            )  # Directly use from args when -h is present
        else:
            modname = validate_modulename(
                pyf_files, args.module_name
            )  # Validate modname when -h is not present
        comline_list += ['-m', modname]  # needed for the rest of scaninputline
    # gh-22819 -- end
    files, options = scaninputline(comline_list)
    auxfuncs.options = options
    capi_maps.load_f2cmap_file(options['f2cmap_file'])
    postlist = callcrackfortran(files, options)
    isusedby = {}
    for plist in postlist:
        if 'use' in plist:
            for u in plist['use'].keys():
                if u not in isusedby:
                    isusedby[u] = []
                isusedby[u].append(plist['name'])
    for plist in postlist:
        module_name = plist['name']
        if plist['block'] == 'python module' and '__user__' in module_name:
            if module_name in isusedby:
                # if not quiet:
                usedby = ','.join(f'"{s}"' for s in isusedby[module_name])
                outmess(
                    f'Skipping Makefile build for module "{module_name}" '
                    f'which is used by {usedby}\n')
    if 'signsfile' in options:
        if options['verbose'] > 1:
            outmess(
                'Stopping. Edit the signature file and then run f2py on the signature file: ')
            outmess(f"{os.path.basename(sys.argv[0])} {options['signsfile']}\n")
        return
    for plist in postlist:
        if plist['block'] != 'python module':
            if 'python module' not in options:
                errmess(
                    'Tip: If your original code is Fortran source then you must use -m option.\n')
            raise TypeError('All blocks must be python module blocks but got %s' % (
                repr(plist['block'])))
    auxfuncs.debugoptions = options['debug']
    f90mod_rules.options = options
    auxfuncs.wrapfuncs = options['wrapfuncs']

    ret = buildmodules(postlist)

    for mn in ret.keys():
        dict_append(ret[mn], {'csrc': fobjcsrc, 'h': fobjhsrc})
    return ret


def filter_files(prefix, suffix, files, remove_prefix=None):
    """
    Filter files by prefix and suffix.
    """
    filtered, rest = [], []
    match = re.compile(prefix + r'.*' + suffix + r'\Z').match
    if remove_prefix:
        ind = len(prefix)
    else:
        ind = 0
    for file in [x.strip() for x in files]:
        if match(file):
            filtered.append(file[ind:])
        else:
            rest.append(file)
    return filtered, rest


def get_prefix(module):
    p = os.path.dirname(os.path.dirname(module.__file__))
    return p


class CombineIncludePaths(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        include_paths_set = set(getattr(namespace, 'include_paths', []) or [])
        if option_string == "--include_paths":
            outmess("Use --include-paths or -I instead of --include_paths which will be removed")
        if option_string in {"--include-paths", "--include_paths"}:
            include_paths_set.update(values.split(':'))
        else:
            include_paths_set.add(values)
        namespace.include_paths = list(include_paths_set)

def f2py_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-I", dest="include_paths", action=CombineIncludePaths)
    parser.add_argument("--include-paths", dest="include_paths", action=CombineIncludePaths)
    parser.add_argument("--include_paths", dest="include_paths", action=CombineIncludePaths)
    parser.add_argument("--freethreading-compatible", dest="ftcompat", action=argparse.BooleanOptionalAction)
    return parser

def get_newer_options(iline):
    iline = (' '.join(iline)).split()
    parser = f2py_parser()
    args, remain = parser.parse_known_args(iline)
    ipaths = args.include_paths
    if args.include_paths is None:
        ipaths = []
    return ipaths, args.ftcompat, remain

def make_f2py_compile_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dep", action="append", dest="dependencies")
    parser.add_argument("--backend", choices=['meson', 'distutils'], default='distutils')
    parser.add_argument("-m", dest="module_name")
    return parser

def preparse_sysargv():
    # To keep backwards bug compatibility, newer flags are handled by argparse,
    # and `sys.argv` is passed to the rest of `f2py` as is.
    parser = make_f2py_compile_parser()

    args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv

    backend_key = args.backend
    if MESON_ONLY_VER and backend_key == 'distutils':
        outmess("Cannot use distutils backend with Python>=3.12,"
                " using meson backend instead.\n")
        backend_key = "meson"

    return {
        "dependencies": args.dependencies or [],
        "backend": backend_key,
        "modulename": args.module_name,
    }

def run_compile():
    """
    Do it all in one call!
    """
    import tempfile

    # Collect dependency flags, preprocess sys.argv
    argy = preparse_sysargv()
    modulename = argy["modulename"]
    if modulename is None:
        modulename = 'untitled'
    dependencies = argy["dependencies"]
    backend_key = argy["backend"]
    build_backend = f2py_build_generator(backend_key)

    i = sys.argv.index('-c')
    del sys.argv[i]

    remove_build_dir = 0
    try:
        i = sys.argv.index('--build-dir')
    except ValueError:
        i = None
    if i is not None:
        build_dir = sys.argv[i + 1]
        del sys.argv[i + 1]
        del sys.argv[i]
    else:
        remove_build_dir = 1
        build_dir = tempfile.mkdtemp()

    _reg1 = re.compile(r'--link-')
    sysinfo_flags = [_m for _m in sys.argv[1:] if _reg1.match(_m)]
    sys.argv = [_m for _m in sys.argv if _m not in sysinfo_flags]
    if sysinfo_flags:
        sysinfo_flags = [f[7:] for f in sysinfo_flags]

    _reg2 = re.compile(
        r'--((no-|)(wrap-functions|lower|freethreading-compatible)|debug-capi|quiet|skip-empty-wrappers)|-include')
    f2py_flags = [_m for _m in sys.argv[1:] if _reg2.match(_m)]
    sys.argv = [_m for _m in sys.argv if _m not in f2py_flags]
    f2py_flags2 = []
    fl = 0
    for a in sys.argv[1:]:
        if a in ['only:', 'skip:']:
            fl = 1
        elif a == ':':
            fl = 0
        if fl or a == ':':
            f2py_flags2.append(a)
    if f2py_flags2 and f2py_flags2[-1] != ':':
        f2py_flags2.append(':')
    f2py_flags.extend(f2py_flags2)
    sys.argv = [_m for _m in sys.argv if _m not in f2py_flags2]
    _reg3 = re.compile(
        r'--((f(90)?compiler(-exec|)|compiler)=|help-compiler)')
    flib_flags = [_m for _m in sys.argv[1:] if _reg3.match(_m)]
    sys.argv = [_m for _m in sys.argv if _m not in flib_flags]
    # TODO: Once distutils is dropped completely, i.e. min_ver >= 3.12, unify into --fflags
    reg_f77_f90_flags = re.compile(r'--f(77|90)flags=')
    reg_distutils_flags = re.compile(r'--((f(77|90)exec|opt|arch)=|(debug|noopt|noarch|help-fcompiler))')
    fc_flags = [_m for _m in sys.argv[1:] if reg_f77_f90_flags.match(_m)]
    distutils_flags = [_m for _m in sys.argv[1:] if reg_distutils_flags.match(_m)]
    if not (MESON_ONLY_VER or backend_key == 'meson'):
        fc_flags.extend(distutils_flags)
    sys.argv = [_m for _m in sys.argv if _m not in (fc_flags + distutils_flags)]

    del_list = []
    for s in flib_flags:
        v = '--fcompiler='
        if s[:len(v)] == v:
            if MESON_ONLY_VER or backend_key == 'meson':
                outmess(
                    "--fcompiler cannot be used with meson,"
                    "set compiler with the FC environment variable\n"
                    )
            else:
                from numpy.distutils import fcompiler
                fcompiler.load_all_fcompiler_classes()
                allowed_keys = list(fcompiler.fcompiler_class.keys())
                nv = ov = s[len(v):].lower()
                if ov not in allowed_keys:
                    vmap = {}  # XXX
                    try:
                        nv = vmap[ov]
                    except KeyError:
                        if ov not in vmap.values():
                            print(f'Unknown vendor: "{s[len(v):]}"')
                    nv = ov
                i = flib_flags.index(s)
                flib_flags[i] = '--fcompiler=' + nv  # noqa: B909
                continue
    for s in del_list:
        i = flib_flags.index(s)
        del flib_flags[i]
    assert len(flib_flags) <= 2, repr(flib_flags)

    _reg5 = re.compile(r'--(verbose)')
    setup_flags = [_m for _m in sys.argv[1:] if _reg5.match(_m)]
    sys.argv = [_m for _m in sys.argv if _m not in setup_flags]

    if '--quiet' in f2py_flags:
        setup_flags.append('--quiet')

    # Ugly filter to remove everything but sources
    sources = sys.argv[1:]
    f2cmapopt = '--f2cmap'
    if f2cmapopt in sys.argv:
        i = sys.argv.index(f2cmapopt)
        f2py_flags.extend(sys.argv[i:i + 2])
        del sys.argv[i + 1], sys.argv[i]
        sources = sys.argv[1:]

    pyf_files, _sources = filter_files("", "[.]pyf([.]src|)", sources)
    sources = pyf_files + _sources
    modulename = validate_modulename(pyf_files, modulename)
    extra_objects, sources = filter_files('', '[.](o|a|so|dylib)', sources)
    library_dirs, sources = filter_files('-L', '', sources, remove_prefix=1)
    libraries, sources = filter_files('-l', '', sources, remove_prefix=1)
    undef_macros, sources = filter_files('-U', '', sources, remove_prefix=1)
    define_macros, sources = filter_files('-D', '', sources, remove_prefix=1)
    for i in range(len(define_macros)):
        name_value = define_macros[i].split('=', 1)
        if len(name_value) == 1:
            name_value.append(None)
        if len(name_value) == 2:
            define_macros[i] = tuple(name_value)
        else:
            print('Invalid use of -D:', name_value)

    # Construct wrappers / signatures / things
    if backend_key == 'meson':
        if not pyf_files:
            outmess('Using meson backend\nWill pass --lower to f2py\nSee https://numpy.org/doc/stable/f2py/buildtools/meson.html\n')
            f2py_flags.append('--lower')
            run_main(f" {' '.join(f2py_flags)} -m {modulename} {' '.join(sources)}".split())
        else:
            run_main(f" {' '.join(f2py_flags)} {' '.join(pyf_files)}".split())

    # Order matters here, includes are needed for run_main above
    include_dirs, _, sources = get_newer_options(sources)
    # Now use the builder
    builder = build_backend(
        modulename,
        sources,
        extra_objects,
        build_dir,
        include_dirs,
        library_dirs,
        libraries,
        define_macros,
        undef_macros,
        f2py_flags,
        sysinfo_flags,
        fc_flags,
        flib_flags,
        setup_flags,
        remove_build_dir,
        {"dependencies": dependencies},
    )

    builder.compile()


def validate_modulename(pyf_files, modulename='untitled'):
    if len(pyf_files) > 1:
        raise ValueError("Only one .pyf file per call")
    if pyf_files:
        pyff = pyf_files[0]
        pyf_modname = auxfuncs.get_f2py_modulename(pyff)
        if modulename != pyf_modname:
            outmess(
                f"Ignoring -m {modulename}.\n"
                f"{pyff} defines {pyf_modname} to be the modulename.\n"
            )
            modulename = pyf_modname
    return modulename

def main():
    if '--help-link' in sys.argv[1:]:
        sys.argv.remove('--help-link')
        if MESON_ONLY_VER:
            outmess("Use --dep for meson builds\n")
        else:
            from numpy.distutils.system_info import show_all
            show_all()
        return

    if '-c' in sys.argv[1:]:
        run_compile()
    else:
        run_main(sys.argv[1:])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\zeta_functions.py ===
""" Riemann zeta and related function. """

from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import ArgumentIndexError, expand_mul, DefinedFunction
from sympy.core.logic import fuzzy_not
from sympy.core.numbers import pi, I, Integer
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.numbers import bernoulli, factorial, genocchi, harmonic
from sympy.functions.elementary.complexes import re, unpolarify, Abs, polar_lift
from sympy.functions.elementary.exponential import log, exp_polar, exp
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys.polytools import Poly

###############################################################################
###################### LERCH TRANSCENDENT #####################################
###############################################################################


class lerchphi(DefinedFunction):
    r"""
    Lerch transcendent (Lerch phi function).

    Explanation
    ===========

    For $\operatorname{Re}(a) > 0$, $|z| < 1$ and $s \in \mathbb{C}$, the
    Lerch transcendent is defined as

    .. math :: \Phi(z, s, a) = \sum_{n=0}^\infty \frac{z^n}{(n + a)^s},

    where the standard branch of the argument is used for $n + a$,
    and by analytic continuation for other values of the parameters.

    A commonly used related function is the Lerch zeta function, defined by

    .. math:: L(q, s, a) = \Phi(e^{2\pi i q}, s, a).

    **Analytic Continuation and Branching Behavior**

    It can be shown that

    .. math:: \Phi(z, s, a) = z\Phi(z, s, a+1) + a^{-s}.

    This provides the analytic continuation to $\operatorname{Re}(a) \le 0$.

    Assume now $\operatorname{Re}(a) > 0$. The integral representation

    .. math:: \Phi_0(z, s, a) = \int_0^\infty \frac{t^{s-1} e^{-at}}{1 - ze^{-t}}
                                \frac{\mathrm{d}t}{\Gamma(s)}

    provides an analytic continuation to $\mathbb{C} - [1, \infty)$.
    Finally, for $x \in (1, \infty)$ we find

    .. math:: \lim_{\epsilon \to 0^+} \Phi_0(x + i\epsilon, s, a)
             -\lim_{\epsilon \to 0^+} \Phi_0(x - i\epsilon, s, a)
             = \frac{2\pi i \log^{s-1}{x}}{x^a \Gamma(s)},

    using the standard branch for both $\log{x}$ and
    $\log{\log{x}}$ (a branch of $\log{\log{x}}$ is needed to
    evaluate $\log{x}^{s-1}$).
    This concludes the analytic continuation. The Lerch transcendent is thus
    branched at $z \in \{0, 1, \infty\}$ and
    $a \in \mathbb{Z}_{\le 0}$. For fixed $z, a$ outside these
    branch points, it is an entire function of $s$.

    Examples
    ========

    The Lerch transcendent is a fairly general function, for this reason it does
    not automatically evaluate to simpler functions. Use ``expand_func()`` to
    achieve this.

    If $z=1$, the Lerch transcendent reduces to the Hurwitz zeta function:

    >>> from sympy import lerchphi, expand_func
    >>> from sympy.abc import z, s, a
    >>> expand_func(lerchphi(1, s, a))
    zeta(s, a)

    More generally, if $z$ is a root of unity, the Lerch transcendent
    reduces to a sum of Hurwitz zeta functions:

    >>> expand_func(lerchphi(-1, s, a))
    zeta(s, a/2)/2**s - zeta(s, a/2 + 1/2)/2**s

    If $a=1$, the Lerch transcendent reduces to the polylogarithm:

    >>> expand_func(lerchphi(z, s, 1))
    polylog(s, z)/z

    More generally, if $a$ is rational, the Lerch transcendent reduces
    to a sum of polylogarithms:

    >>> from sympy import S
    >>> expand_func(lerchphi(z, s, S(1)/2))
    2**(s - 1)*(polylog(s, sqrt(z))/sqrt(z) -
                polylog(s, sqrt(z)*exp_polar(I*pi))/sqrt(z))
    >>> expand_func(lerchphi(z, s, S(3)/2))
    -2**s/z + 2**(s - 1)*(polylog(s, sqrt(z))/sqrt(z) -
                          polylog(s, sqrt(z)*exp_polar(I*pi))/sqrt(z))/z

    The derivatives with respect to $z$ and $a$ can be computed in
    closed form:

    >>> lerchphi(z, s, a).diff(z)
    (-a*lerchphi(z, s, a) + lerchphi(z, s - 1, a))/z
    >>> lerchphi(z, s, a).diff(a)
    -s*lerchphi(z, s + 1, a)

    See Also
    ========

    polylog, zeta

    References
    ==========

    .. [1] Bateman, H.; Erdelyi, A. (1953), Higher Transcendental Functions,
           Vol. I, New York: McGraw-Hill. Section 1.11.
    .. [2] https://dlmf.nist.gov/25.14
    .. [3] https://en.wikipedia.org/wiki/Lerch_transcendent

    """

    def _eval_expand_func(self, **hints):
        z, s, a = self.args
        if z == 1:
            return zeta(s, a)
        if s.is_Integer and s <= 0:
            t = Dummy('t')
            p = Poly((t + a)**(-s), t)
            start = 1/(1 - t)
            res = S.Zero
            for c in reversed(p.all_coeffs()):
                res += c*start
                start = t*start.diff(t)
            return res.subs(t, z)

        if a.is_Rational:
            # See section 18 of
            #   Kelly B. Roach.  Hypergeometric Function Representations.
            #   In: Proceedings of the 1997 International Symposium on Symbolic and
            #   Algebraic Computation, pages 205-211, New York, 1997. ACM.
            # TODO should something be polarified here?
            add = S.Zero
            mul = S.One
            # First reduce a to the interaval (0, 1]
            if a > 1:
                n = floor(a)
                if n == a:
                    n -= 1
                a -= n
                mul = z**(-n)
                add = Add(*[-z**(k - n)/(a + k)**s for k in range(n)])
            elif a <= 0:
                n = floor(-a) + 1
                a += n
                mul = z**n
                add = Add(*[z**(n - 1 - k)/(a - k - 1)**s for k in range(n)])

            m, n = S([a.p, a.q])
            zet = exp_polar(2*pi*I/n)
            root = z**(1/n)
            up_zet = unpolarify(zet)
            addargs = []
            for k in range(n):
                p = polylog(s, zet**k*root)
                if isinstance(p, polylog):
                    p = p._eval_expand_func(**hints)
                addargs.append(p/(up_zet**k*root)**m)
            return add + mul*n**(s - 1)*Add(*addargs)

        # TODO use minpoly instead of ad-hoc methods when issue 5888 is fixed
        if isinstance(z, exp) and (z.args[0]/(pi*I)).is_Rational or z in [-1, I, -I]:
            # TODO reference?
            if z == -1:
                p, q = S([1, 2])
            elif z == I:
                p, q = S([1, 4])
            elif z == -I:
                p, q = S([-1, 4])
            else:
                arg = z.args[0]/(2*pi*I)
                p, q = S([arg.p, arg.q])
            return Add(*[exp(2*pi*I*k*p/q)/q**s*zeta(s, (k + a)/q)
                         for k in range(q)])

        return lerchphi(z, s, a)

    def fdiff(self, argindex=1):
        z, s, a = self.args
        if argindex == 3:
            return -s*lerchphi(z, s + 1, a)
        elif argindex == 1:
            return (lerchphi(z, s - 1, a) - a*lerchphi(z, s, a))/z
        else:
            raise ArgumentIndexError

    def _eval_rewrite_helper(self, target):
        res = self._eval_expand_func()
        if res.has(target):
            return res
        else:
            return self

    def _eval_rewrite_as_zeta(self, z, s, a, **kwargs):
        return self._eval_rewrite_helper(zeta)

    def _eval_rewrite_as_polylog(self, z, s, a, **kwargs):
        return self._eval_rewrite_helper(polylog)

###############################################################################
###################### POLYLOGARITHM ##########################################
###############################################################################


class polylog(DefinedFunction):
    r"""
    Polylogarithm function.

    Explanation
    ===========

    For $|z| < 1$ and $s \in \mathbb{C}$, the polylogarithm is
    defined by

    .. math:: \operatorname{Li}_s(z) = \sum_{n=1}^\infty \frac{z^n}{n^s},

    where the standard branch of the argument is used for $n$. It admits
    an analytic continuation which is branched at $z=1$ (notably not on the
    sheet of initial definition), $z=0$ and $z=\infty$.

    The name polylogarithm comes from the fact that for $s=1$, the
    polylogarithm is related to the ordinary logarithm (see examples), and that

    .. math:: \operatorname{Li}_{s+1}(z) =
                    \int_0^z \frac{\operatorname{Li}_s(t)}{t} \mathrm{d}t.

    The polylogarithm is a special case of the Lerch transcendent:

    .. math:: \operatorname{Li}_{s}(z) = z \Phi(z, s, 1).

    Examples
    ========

    For $z \in \{0, 1, -1\}$, the polylogarithm is automatically expressed
    using other functions:

    >>> from sympy import polylog
    >>> from sympy.abc import s
    >>> polylog(s, 0)
    0
    >>> polylog(s, 1)
    zeta(s)
    >>> polylog(s, -1)
    -dirichlet_eta(s)

    If $s$ is a negative integer, $0$ or $1$, the polylogarithm can be
    expressed using elementary functions. This can be done using
    ``expand_func()``:

    >>> from sympy import expand_func
    >>> from sympy.abc import z
    >>> expand_func(polylog(1, z))
    -log(1 - z)
    >>> expand_func(polylog(0, z))
    z/(1 - z)

    The derivative with respect to $z$ can be computed in closed form:

    >>> polylog(s, z).diff(z)
    polylog(s - 1, z)/z

    The polylogarithm can be expressed in terms of the lerch transcendent:

    >>> from sympy import lerchphi
    >>> polylog(s, z).rewrite(lerchphi)
    z*lerchphi(z, s, 1)

    See Also
    ========

    zeta, lerchphi

    """

    @classmethod
    def eval(cls, s, z):
        if z.is_number:
            if z is S.One:
                return zeta(s)
            elif z is S.NegativeOne:
                return -dirichlet_eta(s)
            elif z is S.Zero:
                return S.Zero
            elif s == 2:
                dilogtable = _dilogtable()
                if z in dilogtable:
                    return dilogtable[z]

        if z.is_zero:
            return S.Zero

        # Make an effort to determine if z is 1 to avoid replacing into
        # expression with singularity
        zone = z.equals(S.One)

        if zone:
            return zeta(s)
        elif zone is False:
            # For s = 0 or -1 use explicit formulas to evaluate, but
            # automatically expanding polylog(1, z) to -log(1-z) seems
            # undesirable for summation methods based on hypergeometric
            # functions
            if s is S.Zero:
                return z/(1 - z)
            elif s is S.NegativeOne:
                return z/(1 - z)**2
            if s.is_zero:
                return z/(1 - z)

        # polylog is branched, but not over the unit disk
        if z.has(exp_polar, polar_lift) and (zone or (Abs(z) <= S.One) == True):
            return cls(s, unpolarify(z))

    def fdiff(self, argindex=1):
        s, z = self.args
        if argindex == 2:
            return polylog(s - 1, z)/z
        raise ArgumentIndexError

    def _eval_rewrite_as_lerchphi(self, s, z, **kwargs):
        return z*lerchphi(z, s, 1)

    def _eval_expand_func(self, **hints):
        s, z = self.args
        if s == 1:
            return -log(1 - z)
        if s.is_Integer and s <= 0:
            u = Dummy('u')
            start = u/(1 - u)
            for _ in range(-s):
                start = u*start.diff(u)
            return expand_mul(start).subs(u, z)
        return polylog(s, z)

    def _eval_is_zero(self):
        z = self.args[1]
        if z.is_zero:
            return True

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import Order
        nu, z = self.args

        z0 = z.subs(x, 0)
        if z0 is S.NaN:
            z0 = z.limit(x, 0, dir='-' if re(cdir).is_negative else '+')

        if z0.is_zero:
            # In case of powers less than 1, number of terms need to be computed
            # separately to avoid repeated callings of _eval_nseries with wrong n
            try:
                _, exp = z.leadterm(x)
            except (ValueError, NotImplementedError):
                return self

            if exp.is_positive:
                newn = ceiling(n/exp)
                o = Order(x**n, x)
                r = z._eval_nseries(x, n, logx, cdir).removeO()
                if r is S.Zero:
                    return o

                term = r
                s = [term]
                for k in range(2, newn):
                    term *= r
                    s.append(term/k**nu)
                return Add(*s) + o

        return super(polylog, self)._eval_nseries(x, n, logx, cdir)

###############################################################################
###################### HURWITZ GENERALIZED ZETA FUNCTION ######################
###############################################################################


class zeta(DefinedFunction):
    r"""
    Hurwitz zeta function (or Riemann zeta function).

    Explanation
    ===========

    For $\operatorname{Re}(a) > 0$ and $\operatorname{Re}(s) > 1$, this
    function is defined as

    .. math:: \zeta(s, a) = \sum_{n=0}^\infty \frac{1}{(n + a)^s},

    where the standard choice of argument for $n + a$ is used. For fixed
    $a$ not a nonpositive integer the Hurwitz zeta function admits a
    meromorphic continuation to all of $\mathbb{C}$; it is an unbranched
    function with a simple pole at $s = 1$.

    The Hurwitz zeta function is a special case of the Lerch transcendent:

    .. math:: \zeta(s, a) = \Phi(1, s, a).

    This formula defines an analytic continuation for all possible values of
    $s$ and $a$ (also $\operatorname{Re}(a) < 0$), see the documentation of
    :class:`lerchphi` for a description of the branching behavior.

    If no value is passed for $a$ a default value of $a = 1$ is assumed,
    yielding the Riemann zeta function.

    Examples
    ========

    For $a = 1$ the Hurwitz zeta function reduces to the famous Riemann
    zeta function:

    .. math:: \zeta(s, 1) = \zeta(s) = \sum_{n=1}^\infty \frac{1}{n^s}.

    >>> from sympy import zeta
    >>> from sympy.abc import s
    >>> zeta(s, 1)
    zeta(s)
    >>> zeta(s)
    zeta(s)

    The Riemann zeta function can also be expressed using the Dirichlet eta
    function:

    >>> from sympy import dirichlet_eta
    >>> zeta(s).rewrite(dirichlet_eta)
    dirichlet_eta(s)/(1 - 2**(1 - s))

    The Riemann zeta function at nonnegative even and negative integer
    values is related to the Bernoulli numbers and polynomials:

    >>> zeta(2)
    pi**2/6
    >>> zeta(4)
    pi**4/90
    >>> zeta(0)
    -1/2
    >>> zeta(-1)
    -1/12
    >>> zeta(-4)
    0

    The specific formulae are:

    .. math:: \zeta(2n) = -\frac{(2\pi i)^{2n} B_{2n}}{2(2n)!}
    .. math:: \zeta(-n,a) = -\frac{B_{n+1}(a)}{n+1}

    No closed-form expressions are known at positive odd integers, but
    numerical evaluation is possible:

    >>> zeta(3).n()
    1.20205690315959

    The derivative of $\zeta(s, a)$ with respect to $a$ can be computed:

    >>> from sympy.abc import a
    >>> zeta(s, a).diff(a)
    -s*zeta(s + 1, a)

    However the derivative with respect to $s$ has no useful closed form
    expression:

    >>> zeta(s, a).diff(s)
    Derivative(zeta(s, a), s)

    The Hurwitz zeta function can be expressed in terms of the Lerch
    transcendent, :class:`~.lerchphi`:

    >>> from sympy import lerchphi
    >>> zeta(s, a).rewrite(lerchphi)
    lerchphi(1, s, a)

    See Also
    ========

    dirichlet_eta, lerchphi, polylog

    References
    ==========

    .. [1] https://dlmf.nist.gov/25.11
    .. [2] https://en.wikipedia.org/wiki/Hurwitz_zeta_function

    """

    @classmethod
    def eval(cls, s, a=None):
        if a is S.One:
            return cls(s)
        elif s is S.NaN or a is S.NaN:
            return S.NaN
        elif s is S.One:
            return S.ComplexInfinity
        elif s is S.Infinity:
            return S.One
        elif a is S.Infinity:
            return S.Zero

        sint = s.is_Integer
        if a is None:
            a = S.One
        if sint and s.is_nonpositive:
            return bernoulli(1-s, a) / (s-1)
        elif a is S.One:
            if sint and s.is_even:
                return -(2*pi*I)**s * bernoulli(s) / (2*factorial(s))
        elif sint and a.is_Integer and a.is_positive:
            return cls(s) - harmonic(a-1, s)
        elif a.is_Integer and a.is_nonpositive and \
                (s.is_integer is False or s.is_nonpositive is False):
            return S.NaN

    def _eval_rewrite_as_bernoulli(self, s, a=1, **kwargs):
        if a == 1 and s.is_integer and s.is_nonnegative and s.is_even:
            return -(2*pi*I)**s * bernoulli(s) / (2*factorial(s))
        return bernoulli(1-s, a) / (s-1)

    def _eval_rewrite_as_dirichlet_eta(self, s, a=1, **kwargs):
        if a != 1:
            return self
        s = self.args[0]
        return dirichlet_eta(s)/(1 - 2**(1 - s))

    def _eval_rewrite_as_lerchphi(self, s, a=1, **kwargs):
        return lerchphi(1, s, a)

    def _eval_is_finite(self):
        return fuzzy_not((self.args[0] - 1).is_zero)

    def _eval_expand_func(self, **hints):
        s = self.args[0]
        a = self.args[1] if len(self.args) > 1 else S.One
        if a.is_integer:
            if a.is_positive:
                return zeta(s) - harmonic(a-1, s)
            if a.is_nonpositive and (s.is_integer is False or
                    s.is_nonpositive is False):
                return S.NaN
        return self

    def fdiff(self, argindex=1):
        if len(self.args) == 2:
            s, a = self.args
        else:
            s, a = self.args + (1,)
        if argindex == 2:
            return -s*zeta(s + 1, a)
        else:
            raise ArgumentIndexError

    def _eval_as_leading_term(self, x, logx, cdir):
        if len(self.args) == 2:
            s, a = self.args
        else:
            s, a = self.args + (S.One,)

        try:
            c, e = a.leadterm(x)
        except NotImplementedError:
            return self

        if e.is_negative and not s.is_positive:
            raise NotImplementedError

        return super(zeta, self)._eval_as_leading_term(x, logx=logx, cdir=cdir)


class dirichlet_eta(DefinedFunction):
    r"""
    Dirichlet eta function.

    Explanation
    ===========

    For $\operatorname{Re}(s) > 0$ and $0 < x \le 1$, this function is defined as

    .. math:: \eta(s, a) = \sum_{n=0}^\infty \frac{(-1)^n}{(n+a)^s}.

    It admits a unique analytic continuation to all of $\mathbb{C}$ for any
    fixed $a$ not a nonpositive integer. It is an entire, unbranched function.

    It can be expressed using the Hurwitz zeta function as

    .. math:: \eta(s, a) = \zeta(s,a) - 2^{1-s} \zeta\left(s, \frac{a+1}{2}\right)

    and using the generalized Genocchi function as

    .. math:: \eta(s, a) = \frac{G(1-s, a)}{2(s-1)}.

    In both cases the limiting value of $\log2 - \psi(a) + \psi\left(\frac{a+1}{2}\right)$
    is used when $s = 1$.

    Examples
    ========

    >>> from sympy import dirichlet_eta, zeta
    >>> from sympy.abc import s
    >>> dirichlet_eta(s).rewrite(zeta)
    Piecewise((log(2), Eq(s, 1)), ((1 - 2**(1 - s))*zeta(s), True))

    See Also
    ========

    zeta

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Dirichlet_eta_function
    .. [2] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743

    """

    @classmethod
    def eval(cls, s, a=None):
        if a is S.One:
            return cls(s)
        if a is None:
            if s == 1:
                return log(2)
            z = zeta(s)
            if not z.has(zeta):
                return (1 - 2**(1-s)) * z
            return
        elif s == 1:
            from sympy.functions.special.gamma_functions import digamma
            return log(2) - digamma(a) + digamma((a+1)/2)
        z1 = zeta(s, a)
        z2 = zeta(s, (a+1)/2)
        if not z1.has(zeta) and not z2.has(zeta):
            return z1 - 2**(1-s) * z2

    def _eval_rewrite_as_zeta(self, s, a=1, **kwargs):
        from sympy.functions.special.gamma_functions import digamma
        if a == 1:
            return Piecewise((log(2), Eq(s, 1)), ((1 - 2**(1-s)) * zeta(s), True))
        return Piecewise((log(2) - digamma(a) + digamma((a+1)/2), Eq(s, 1)),
                (zeta(s, a) - 2**(1-s) * zeta(s, (a+1)/2), True))

    def _eval_rewrite_as_genocchi(self, s, a=S.One, **kwargs):
        from sympy.functions.special.gamma_functions import digamma
        return Piecewise((log(2) - digamma(a) + digamma((a+1)/2), Eq(s, 1)),
                (genocchi(1-s, a) / (2 * (s-1)), True))

    def _eval_evalf(self, prec):
        if all(i.is_number for i in self.args):
            return self.rewrite(zeta)._eval_evalf(prec)


class riemann_xi(DefinedFunction):
    r"""
    Riemann Xi function.

    Examples
    ========

    The Riemann Xi function is closely related to the Riemann zeta function.
    The zeros of Riemann Xi function are precisely the non-trivial zeros
    of the zeta function.

    >>> from sympy import riemann_xi, zeta
    >>> from sympy.abc import s
    >>> riemann_xi(s).rewrite(zeta)
    s*(s - 1)*gamma(s/2)*zeta(s)/(2*pi**(s/2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Riemann_Xi_function

    """


    @classmethod
    def eval(cls, s):
        from sympy.functions.special.gamma_functions import gamma
        z = zeta(s)
        if s in (S.Zero, S.One):
            return S.Half

        if not isinstance(z, zeta):
            return s*(s - 1)*gamma(s/2)*z/(2*pi**(s/2))

    def _eval_rewrite_as_zeta(self, s, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        return s*(s - 1)*gamma(s/2)*zeta(s)/(2*pi**(s/2))


class stieltjes(DefinedFunction):
    r"""
    Represents Stieltjes constants, $\gamma_{k}$ that occur in
    Laurent Series expansion of the Riemann zeta function.

    Examples
    ========

    >>> from sympy import stieltjes
    >>> from sympy.abc import n, m
    >>> stieltjes(n)
    stieltjes(n)

    The zero'th stieltjes constant:

    >>> stieltjes(0)
    EulerGamma
    >>> stieltjes(0, 1)
    EulerGamma

    For generalized stieltjes constants:

    >>> stieltjes(n, m)
    stieltjes(n, m)

    Constants are only defined for integers >= 0:

    >>> stieltjes(-1)
    zoo

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Stieltjes_constants

    """

    @classmethod
    def eval(cls, n, a=None):
        if a is not None:
            a = sympify(a)
            if a is S.NaN:
                return S.NaN
            if a.is_Integer and a.is_nonpositive:
                return S.ComplexInfinity

        if n.is_Number:
            if n is S.NaN:
                return S.NaN
            elif n < 0:
                return S.ComplexInfinity
            elif not n.is_Integer:
                return S.ComplexInfinity
            elif n is S.Zero and a in [None, 1]:
                return S.EulerGamma

        if n.is_extended_negative:
            return S.ComplexInfinity

        if n.is_zero and a in [None, 1]:
            return S.EulerGamma

        if n.is_integer == False:
            return S.ComplexInfinity


@cacheit
def _dilogtable():
    return {
        S.Half: pi**2/12 - log(2)**2/2,
        Integer(2) : pi**2/4 - I*pi*log(2),
        -(sqrt(5) - 1)/2 : -pi**2/15 + log((sqrt(5)-1)/2)**2/2,
        -(sqrt(5) + 1)/2 : -pi**2/10 - log((sqrt(5)+1)/2)**2,
        (3 - sqrt(5))/2 : pi**2/15 - log((sqrt(5)-1)/2)**2,
        (sqrt(5) - 1)/2 : pi**2/10 - log((sqrt(5)-1)/2)**2,
        I : I*S.Catalan - pi**2/48,
        -I : -I*S.Catalan - pi**2/48,
        1 - I : pi**2/16 - I*S.Catalan - pi*I/4*log(2),
        1 + I : pi**2/16 + I*S.Catalan + pi*I/4*log(2),
        (1 - I)/2 : -log(2)**2/8 + pi*I*log(2)/8 + 5*pi**2/96 - I*S.Catalan
    }

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\numberfields\primes.py ===
"""Prime ideals in number fields. """

from sympy.polys.polytools import Poly
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable
from sympy.utilities.decorator import public
from .basis import round_two, nilradical_mod_p
from .exceptions import StructureError
from .modules import ModuleEndomorphism, find_min_poly
from .utilities import coeff_search, supplement_a_subspace


def _check_formal_conditions_for_maximal_order(submodule):
    r"""
    Several functions in this module accept an argument which is to be a
    :py:class:`~.Submodule` representing the maximal order in a number field,
    such as returned by the :py:func:`~sympy.polys.numberfields.basis.round_two`
    algorithm.

    We do not attempt to check that the given ``Submodule`` actually represents
    a maximal order, but we do check a basic set of formal conditions that the
    ``Submodule`` must satisfy, at a minimum. The purpose is to catch an
    obviously ill-formed argument.
    """
    prefix = 'The submodule representing the maximal order should '
    cond = None
    if not submodule.is_power_basis_submodule():
        cond = 'be a direct submodule of a power basis.'
    elif not submodule.starts_with_unity():
        cond = 'have 1 as its first generator.'
    elif not submodule.is_sq_maxrank_HNF():
        cond = 'have square matrix, of maximal rank, in Hermite Normal Form.'
    if cond is not None:
        raise StructureError(prefix + cond)


class PrimeIdeal(IntegerPowerable):
    r"""
    A prime ideal in a ring of algebraic integers.
    """

    def __init__(self, ZK, p, alpha, f, e=None):
        """
        Parameters
        ==========

        ZK : :py:class:`~.Submodule`
            The maximal order where this ideal lives.
        p : int
            The rational prime this ideal divides.
        alpha : :py:class:`~.PowerBasisElement`
            Such that the ideal is equal to ``p*ZK + alpha*ZK``.
        f : int
            The inertia degree.
        e : int, ``None``, optional
            The ramification index, if already known. If ``None``, we will
            compute it here.

        """
        _check_formal_conditions_for_maximal_order(ZK)
        self.ZK = ZK
        self.p = p
        self.alpha = alpha
        self.f = f
        self._test_factor = None
        self.e = e if e is not None else self.valuation(p * ZK)

    def __str__(self):
        if self.is_inert:
            return f'({self.p})'
        return f'({self.p}, {self.alpha.as_expr()})'

    @property
    def is_inert(self):
        """
        Say whether the rational prime we divide is inert, i.e. stays prime in
        our ring of integers.
        """
        return self.f == self.ZK.n

    def repr(self, field_gen=None, just_gens=False):
        """
        Print a representation of this prime ideal.

        Examples
        ========

        >>> from sympy import cyclotomic_poly, QQ
        >>> from sympy.abc import x, zeta
        >>> T = cyclotomic_poly(7, x)
        >>> K = QQ.algebraic_field((T, zeta))
        >>> P = K.primes_above(11)
        >>> print(P[0].repr())
        [ (11, x**3 + 5*x**2 + 4*x - 1) e=1, f=3 ]
        >>> print(P[0].repr(field_gen=zeta))
        [ (11, zeta**3 + 5*zeta**2 + 4*zeta - 1) e=1, f=3 ]
        >>> print(P[0].repr(field_gen=zeta, just_gens=True))
        (11, zeta**3 + 5*zeta**2 + 4*zeta - 1)

        Parameters
        ==========

        field_gen : :py:class:`~.Symbol`, ``None``, optional (default=None)
            The symbol to use for the generator of the field. This will appear
            in our representation of ``self.alpha``. If ``None``, we use the
            variable of the defining polynomial of ``self.ZK``.
        just_gens : bool, optional (default=False)
            If ``True``, just print the "(p, alpha)" part, showing "just the
            generators" of the prime ideal. Otherwise, print a string of the
            form "[ (p, alpha) e=..., f=... ]", giving the ramification index
            and inertia degree, along with the generators.

        """
        field_gen = field_gen or self.ZK.parent.T.gen
        p, alpha, e, f = self.p, self.alpha, self.e, self.f
        alpha_rep = str(alpha.numerator(x=field_gen).as_expr())
        if alpha.denom > 1:
            alpha_rep = f'({alpha_rep})/{alpha.denom}'
        gens = f'({p}, {alpha_rep})'
        if just_gens:
            return gens
        return f'[ {gens} e={e}, f={f} ]'

    def __repr__(self):
        return self.repr()

    def as_submodule(self):
        r"""
        Represent this prime ideal as a :py:class:`~.Submodule`.

        Explanation
        ===========

        The :py:class:`~.PrimeIdeal` class serves to bundle information about
        a prime ideal, such as its inertia degree, ramification index, and
        two-generator representation, as well as to offer helpful methods like
        :py:meth:`~.PrimeIdeal.valuation` and
        :py:meth:`~.PrimeIdeal.test_factor`.

        However, in order to be added and multiplied by other ideals or
        rational numbers, it must first be converted into a
        :py:class:`~.Submodule`, which is a class that supports these
        operations.

        In many cases, the user need not perform this conversion deliberately,
        since it is automatically performed by the arithmetic operator methods
        :py:meth:`~.PrimeIdeal.__add__` and :py:meth:`~.PrimeIdeal.__mul__`.

        Raising a :py:class:`~.PrimeIdeal` to a non-negative integer power is
        also supported.

        Examples
        ========

        >>> from sympy import Poly, cyclotomic_poly, prime_decomp
        >>> T = Poly(cyclotomic_poly(7))
        >>> P0 = prime_decomp(7, T)[0]
        >>> print(P0**6 == 7*P0.ZK)
        True

        Note that, on both sides of the equation above, we had a
        :py:class:`~.Submodule`. In the next equation we recall that adding
        ideals yields their GCD. This time, we need a deliberate conversion
        to :py:class:`~.Submodule` on the right:

        >>> print(P0 + 7*P0.ZK == P0.as_submodule())
        True

        Returns
        =======

        :py:class:`~.Submodule`
            Will be equal to ``self.p * self.ZK + self.alpha * self.ZK``.

        See Also
        ========

        __add__
        __mul__

        """
        M = self.p * self.ZK + self.alpha * self.ZK
        # Pre-set expensive boolean properties whose value we already know:
        M._starts_with_unity = False
        M._is_sq_maxrank_HNF = True
        return M

    def __eq__(self, other):
        if isinstance(other, PrimeIdeal):
            return self.as_submodule() == other.as_submodule()
        return NotImplemented

    def __add__(self, other):
        """
        Convert to a :py:class:`~.Submodule` and add to another
        :py:class:`~.Submodule`.

        See Also
        ========

        as_submodule

        """
        return self.as_submodule() + other

    __radd__ = __add__

    def __mul__(self, other):
        """
        Convert to a :py:class:`~.Submodule` and multiply by another
        :py:class:`~.Submodule` or a rational number.

        See Also
        ========

        as_submodule

        """
        return self.as_submodule() * other

    __rmul__ = __mul__

    def _zeroth_power(self):
        return self.ZK

    def _first_power(self):
        return self

    def test_factor(self):
        r"""
        Compute a test factor for this prime ideal.

        Explanation
        ===========

        Write $\mathfrak{p}$ for this prime ideal, $p$ for the rational prime
        it divides. Then, for computing $\mathfrak{p}$-adic valuations it is
        useful to have a number $\beta \in \mathbb{Z}_K$ such that
        $p/\mathfrak{p} = p \mathbb{Z}_K + \beta \mathbb{Z}_K$.

        Essentially, this is the same as the number $\Psi$ (or the "reagent")
        from Kummer's 1847 paper (*Ueber die Zerlegung...*, Crelle vol. 35) in
        which ideal divisors were invented.
        """
        if self._test_factor is None:
            self._test_factor = _compute_test_factor(self.p, [self.alpha], self.ZK)
        return self._test_factor

    def valuation(self, I):
        r"""
        Compute the $\mathfrak{p}$-adic valuation of integral ideal I at this
        prime ideal.

        Parameters
        ==========

        I : :py:class:`~.Submodule`

        See Also
        ========

        prime_valuation

        """
        return prime_valuation(I, self)

    def reduce_element(self, elt):
        """
        Reduce a :py:class:`~.PowerBasisElement` to a "small representative"
        modulo this prime ideal.

        Parameters
        ==========

        elt : :py:class:`~.PowerBasisElement`
            The element to be reduced.

        Returns
        =======

        :py:class:`~.PowerBasisElement`
            The reduced element.

        See Also
        ========

        reduce_ANP
        reduce_alg_num
        .Submodule.reduce_element

        """
        return self.as_submodule().reduce_element(elt)

    def reduce_ANP(self, a):
        """
        Reduce an :py:class:`~.ANP` to a "small representative" modulo this
        prime ideal.

        Parameters
        ==========

        elt : :py:class:`~.ANP`
            The element to be reduced.

        Returns
        =======

        :py:class:`~.ANP`
            The reduced element.

        See Also
        ========

        reduce_element
        reduce_alg_num
        .Submodule.reduce_element

        """
        elt = self.ZK.parent.element_from_ANP(a)
        red = self.reduce_element(elt)
        return red.to_ANP()

    def reduce_alg_num(self, a):
        """
        Reduce an :py:class:`~.AlgebraicNumber` to a "small representative"
        modulo this prime ideal.

        Parameters
        ==========

        elt : :py:class:`~.AlgebraicNumber`
            The element to be reduced.

        Returns
        =======

        :py:class:`~.AlgebraicNumber`
            The reduced element.

        See Also
        ========

        reduce_element
        reduce_ANP
        .Submodule.reduce_element

        """
        elt = self.ZK.parent.element_from_alg_num(a)
        red = self.reduce_element(elt)
        return a.field_element(list(reversed(red.QQ_col.flat())))


def _compute_test_factor(p, gens, ZK):
    r"""
    Compute the test factor for a :py:class:`~.PrimeIdeal` $\mathfrak{p}$.

    Parameters
    ==========

    p : int
        The rational prime $\mathfrak{p}$ divides

    gens : list of :py:class:`PowerBasisElement`
        A complete set of generators for $\mathfrak{p}$ over *ZK*, EXCEPT that
        an element equivalent to rational *p* can and should be omitted (since
        it has no effect except to waste time).

    ZK : :py:class:`~.Submodule`
        The maximal order where the prime ideal $\mathfrak{p}$ lives.

    Returns
    =======

    :py:class:`~.PowerBasisElement`

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
    (See Proposition 4.8.15.)

    """
    _check_formal_conditions_for_maximal_order(ZK)
    E = ZK.endomorphism_ring()
    matrices = [E.inner_endomorphism(g).matrix(modulus=p) for g in gens]
    B = DomainMatrix.zeros((0, ZK.n), FF(p)).vstack(*matrices)
    # A nonzero element of the nullspace of B will represent a
    # lin comb over the omegas which (i) is not a multiple of p
    # (since it is nonzero over FF(p)), while (ii) is such that
    # its product with each g in gens _is_ a multiple of p (since
    # B represents multiplication by these generators). Theory
    # predicts that such an element must exist, so nullspace should
    # be non-trivial.
    x = B.nullspace()[0, :].transpose()
    beta = ZK.parent(ZK.matrix * x.convert_to(ZZ), denom=ZK.denom)
    return beta


@public
def prime_valuation(I, P):
    r"""
    Compute the *P*-adic valuation for an integral ideal *I*.

    Examples
    ========

    >>> from sympy import QQ
    >>> from sympy.polys.numberfields import prime_valuation
    >>> K = QQ.cyclotomic_field(5)
    >>> P = K.primes_above(5)
    >>> ZK = K.maximal_order()
    >>> print(prime_valuation(25*ZK, P[0]))
    8

    Parameters
    ==========

    I : :py:class:`~.Submodule`
        An integral ideal whose valuation is desired.

    P : :py:class:`~.PrimeIdeal`
        The prime at which to compute the valuation.

    Returns
    =======

    int

    See Also
    ========

    .PrimeIdeal.valuation

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithm 4.8.17.)

    """
    p, ZK = P.p, P.ZK
    n, W, d = ZK.n, ZK.matrix, ZK.denom

    A = W.convert_to(QQ).inv() * I.matrix * d / I.denom
    # Although A must have integer entries, given that I is an integral ideal,
    # as a DomainMatrix it will still be over QQ, so we convert back:
    A = A.convert_to(ZZ)
    D = A.det()
    if D % p != 0:
        return 0

    beta = P.test_factor()

    f = d ** n // W.det()
    need_complete_test = (f % p == 0)
    v = 0
    while True:
        # Entering the loop, the cols of A represent lin combs of omegas.
        # Turn them into lin combs of thetas:
        A = W * A
        # And then one column at a time...
        for j in range(n):
            c = ZK.parent(A[:, j], denom=d)
            c *= beta
            # ...turn back into lin combs of omegas, after multiplying by beta:
            c = ZK.represent(c).flat()
            for i in range(n):
                A[i, j] = c[i]
        if A[n - 1, n - 1].element % p != 0:
            break
        A = A / p
        # As noted above, domain converts to QQ even when division goes evenly.
        # So must convert back, even when we don't "need_complete_test".
        if need_complete_test:
            # In this case, having a non-integer entry is actually just our
            # halting condition.
            try:
                A = A.convert_to(ZZ)
            except CoercionFailed:
                break
        else:
            # In this case theory says we should not have any non-integer entries.
            A = A.convert_to(ZZ)
        v += 1
    return v


def _two_elt_rep(gens, ZK, p, f=None, Np=None):
    r"""
    Given a set of *ZK*-generators of a prime ideal, compute a set of just two
    *ZK*-generators for the same ideal, one of which is *p* itself.

    Parameters
    ==========

    gens : list of :py:class:`PowerBasisElement`
        Generators for the prime ideal over *ZK*, the ring of integers of the
        field $K$.

    ZK : :py:class:`~.Submodule`
        The maximal order in $K$.

    p : int
        The rational prime divided by the prime ideal.

    f : int, optional
        The inertia degree of the prime ideal, if known.

    Np : int, optional
        The norm $p^f$ of the prime ideal, if known.
        NOTE: There is no reason to supply both *f* and *Np*. Either one will
        save us from having to compute the norm *Np* ourselves. If both are known,
        *Np* is preferred since it saves one exponentiation.

    Returns
    =======

    :py:class:`~.PowerBasisElement` representing a single algebraic integer
    alpha such that the prime ideal is equal to ``p*ZK + alpha*ZK``.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
    (See Algorithm 4.7.10.)

    """
    _check_formal_conditions_for_maximal_order(ZK)
    pb = ZK.parent
    T = pb.T
    # Detect the special cases in which either (a) all generators are multiples
    # of p, or (b) there are no generators (so `all` is vacuously true):
    if all((g % p).equiv(0) for g in gens):
        return pb.zero()

    if Np is None:
        if f is not None:
            Np = p**f
        else:
            Np = abs(pb.submodule_from_gens(gens).matrix.det())

    omega = ZK.basis_element_pullbacks()
    beta = [p*om for om in omega[1:]]  # note: we omit omega[0] == 1
    beta += gens
    search = coeff_search(len(beta), 1)
    for c in search:
        alpha = sum(ci*betai for ci, betai in zip(c, beta))
        # Note: It may be tempting to reduce alpha mod p here, to try to work
        # with smaller numbers, but must not do that, as it can result in an
        # infinite loop! E.g. try factoring 2 in Q(sqrt(-7)).
        n = alpha.norm(T) // Np
        if n % p != 0:
            # Now can reduce alpha mod p.
            return alpha % p


def _prime_decomp_easy_case(p, ZK):
    r"""
    Compute the decomposition of rational prime *p* in the ring of integers
    *ZK* (given as a :py:class:`~.Submodule`), in the "easy case", i.e. the
    case where *p* does not divide the index of $\theta$ in *ZK*, where
    $\theta$ is the generator of the ``PowerBasis`` of which *ZK* is a
    ``Submodule``.
    """
    T = ZK.parent.T
    T_bar = Poly(T, modulus=p)
    lc, fl = T_bar.factor_list()
    if len(fl) == 1 and fl[0][1] == 1:
        return [PrimeIdeal(ZK, p, ZK.parent.zero(), ZK.n, 1)]
    return [PrimeIdeal(ZK, p,
                       ZK.parent.element_from_poly(Poly(t, domain=ZZ)),
                       t.degree(), e)
            for t, e in fl]


def _prime_decomp_compute_kernel(I, p, ZK):
    r"""
    Parameters
    ==========

    I : :py:class:`~.Module`
        An ideal of ``ZK/pZK``.
    p : int
        The rational prime being factored.
    ZK : :py:class:`~.Submodule`
        The maximal order.

    Returns
    =======

    Pair ``(N, G)``, where:

        ``N`` is a :py:class:`~.Module` representing the kernel of the map
        ``a |--> a**p - a`` on ``(O/pO)/I``, guaranteed to be a module with
        unity.

        ``G`` is a :py:class:`~.Module` representing a basis for the separable
        algebra ``A = O/I`` (see Cohen).

    """
    W = I.matrix
    n, r = W.shape
    # Want to take the Fp-basis given by the columns of I, adjoin (1, 0, ..., 0)
    # (which we know is not already in there since I is a basis for a prime ideal)
    # and then supplement this with additional columns to make an invertible n x n
    # matrix. This will then represent a full basis for ZK, whose first r columns
    # are pullbacks of the basis for I.
    if r == 0:
        B = W.eye(n, ZZ)
    else:
        B = W.hstack(W.eye(n, ZZ)[:, 0])
    if B.shape[1] < n:
        B = supplement_a_subspace(B.convert_to(FF(p))).convert_to(ZZ)

    G = ZK.submodule_from_matrix(B)
    # Must compute G's multiplication table _before_ discarding the first r
    # columns. (See Step 9 in Alg 6.2.9 in Cohen, where the betas are actually
    # needed in order to represent each product of gammas. However, once we've
    # found the representations, then we can ignore the betas.)
    G.compute_mult_tab()
    G = G.discard_before(r)

    phi = ModuleEndomorphism(G, lambda x: x**p - x)
    N = phi.kernel(modulus=p)
    assert N.starts_with_unity()
    return N, G


def _prime_decomp_maximal_ideal(I, p, ZK):
    r"""
    We have reached the case where we have a maximal (hence prime) ideal *I*,
    which we know because the quotient ``O/I`` is a field.

    Parameters
    ==========

    I : :py:class:`~.Module`
        An ideal of ``O/pO``.
    p : int
        The rational prime being factored.
    ZK : :py:class:`~.Submodule`
        The maximal order.

    Returns
    =======

    :py:class:`~.PrimeIdeal` instance representing this prime

    """
    m, n = I.matrix.shape
    f = m - n
    G = ZK.matrix * I.matrix
    gens = [ZK.parent(G[:, j], denom=ZK.denom) for j in range(G.shape[1])]
    alpha = _two_elt_rep(gens, ZK, p, f=f)
    return PrimeIdeal(ZK, p, alpha, f)


def _prime_decomp_split_ideal(I, p, N, G, ZK):
    r"""
    Perform the step in the prime decomposition algorithm where we have determined
    the quotient ``ZK/I`` is _not_ a field, and we want to perform a non-trivial
    factorization of *I* by locating an idempotent element of ``ZK/I``.
    """
    assert I.parent == ZK and G.parent is ZK and N.parent is G
    # Since ZK/I is not a field, the kernel computed in the previous step contains
    # more than just the prime field Fp, and our basis N for the nullspace therefore
    # contains at least a second column (which represents an element outside Fp).
    # Let alpha be such an element:
    alpha = N(1).to_parent()
    assert alpha.module is G

    alpha_powers = []
    m = find_min_poly(alpha, FF(p), powers=alpha_powers)
    # TODO (future work):
    #  We don't actually need full factorization, so might use a faster method
    #  to just break off a single non-constant factor m1?
    lc, fl = m.factor_list()
    m1 = fl[0][0]
    m2 = m.quo(m1)
    U, V, g = m1.gcdex(m2)
    # Sanity check: theory says m is squarefree, so m1, m2 should be coprime:
    assert g == 1
    E = list(reversed(Poly(U * m1, domain=ZZ).rep.to_list()))
    eps1 = sum(E[i]*alpha_powers[i] for i in range(len(E)))
    eps2 = 1 - eps1
    idemps = [eps1, eps2]
    factors = []
    for eps in idemps:
        e = eps.to_parent()
        assert e.module is ZK
        D = I.matrix.convert_to(FF(p)).hstack(*[
            (e * om).column(domain=FF(p)) for om in ZK.basis_elements()
        ])
        W = D.columnspace().convert_to(ZZ)
        H = ZK.submodule_from_matrix(W)
        factors.append(H)
    return factors


@public
def prime_decomp(p, T=None, ZK=None, dK=None, radical=None):
    r"""
    Compute the decomposition of rational prime *p* in a number field.

    Explanation
    ===========

    Ordinarily this should be accessed through the
    :py:meth:`~.AlgebraicField.primes_above` method of an
    :py:class:`~.AlgebraicField`.

    Examples
    ========

    >>> from sympy import Poly, QQ
    >>> from sympy.abc import x, theta
    >>> T = Poly(x ** 3 + x ** 2 - 2 * x + 8)
    >>> K = QQ.algebraic_field((T, theta))
    >>> print(K.primes_above(2))
    [[ (2, x**2 + 1) e=1, f=1 ], [ (2, (x**2 + 3*x + 2)/2) e=1, f=1 ],
     [ (2, (3*x**2 + 3*x)/2) e=1, f=1 ]]

    Parameters
    ==========

    p : int
        The rational prime whose decomposition is desired.

    T : :py:class:`~.Poly`, optional
        Monic irreducible polynomial defining the number field $K$ in which to
        factor. NOTE: at least one of *T* or *ZK* must be provided.

    ZK : :py:class:`~.Submodule`, optional
        The maximal order for $K$, if already known.
        NOTE: at least one of *T* or *ZK* must be provided.

    dK : int, optional
        The discriminant of the field $K$, if already known.

    radical : :py:class:`~.Submodule`, optional
        The nilradical mod *p* in the integers of $K$, if already known.

    Returns
    =======

    List of :py:class:`~.PrimeIdeal` instances.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithm 6.2.9.)

    """
    if T is None and ZK is None:
        raise ValueError('At least one of T or ZK must be provided.')
    if ZK is not None:
        _check_formal_conditions_for_maximal_order(ZK)
    if T is None:
        T = ZK.parent.T
    radicals = {}
    if dK is None or ZK is None:
        ZK, dK = round_two(T, radicals=radicals)
    dT = T.discriminant()
    f_squared = dT // dK
    if f_squared % p != 0:
        return _prime_decomp_easy_case(p, ZK)
    radical = radical or radicals.get(p) or nilradical_mod_p(ZK, p)
    stack = [radical]
    primes = []
    while stack:
        I = stack.pop()
        N, G = _prime_decomp_compute_kernel(I, p, ZK)
        if N.n == 1:
            P = _prime_decomp_maximal_ideal(I, p, ZK)
            primes.append(P)
        else:
            I1, I2 = _prime_decomp_split_ideal(I, p, N, G, ZK)
            stack.extend([I1, I2])
    return primes

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\postgresql\psycopg.py ===
# dialects/postgresql/psycopg.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

r"""
.. dialect:: postgresql+psycopg
    :name: psycopg (a.k.a. psycopg 3)
    :dbapi: psycopg
    :connectstring: postgresql+psycopg://user:password@host:port/dbname[?key=value&key=value...]
    :url: https://pypi.org/project/psycopg/

``psycopg`` is the package and module name for version 3 of the ``psycopg``
database driver, formerly known as ``psycopg2``.  This driver is different
enough from its ``psycopg2`` predecessor that SQLAlchemy supports it
via a totally separate dialect; support for ``psycopg2`` is expected to remain
for as long as that package continues to function for modern Python versions,
and also remains the default dialect for the ``postgresql://`` dialect
series.

The SQLAlchemy ``psycopg`` dialect provides both a sync and an async
implementation under the same dialect name. The proper version is
selected depending on how the engine is created:

* calling :func:`_sa.create_engine` with ``postgresql+psycopg://...`` will
  automatically select the sync version, e.g.::

    from sqlalchemy import create_engine

    sync_engine = create_engine(
        "postgresql+psycopg://scott:tiger@localhost/test"
    )

* calling :func:`_asyncio.create_async_engine` with
  ``postgresql+psycopg://...`` will automatically select the async version,
  e.g.::

    from sqlalchemy.ext.asyncio import create_async_engine

    asyncio_engine = create_async_engine(
        "postgresql+psycopg://scott:tiger@localhost/test"
    )

The asyncio version of the dialect may also be specified explicitly using the
``psycopg_async`` suffix, as::

    from sqlalchemy.ext.asyncio import create_async_engine

    asyncio_engine = create_async_engine(
        "postgresql+psycopg_async://scott:tiger@localhost/test"
    )

.. seealso::

    :ref:`postgresql_psycopg2` - The SQLAlchemy ``psycopg``
    dialect shares most of its behavior with the ``psycopg2`` dialect.
    Further documentation is available there.

Using a different Cursor class
------------------------------

One of the differences between ``psycopg`` and the older ``psycopg2``
is how bound parameters are handled: ``psycopg2`` would bind them
client side, while ``psycopg`` by default will bind them server side.

It's possible to configure ``psycopg`` to do client side binding by
specifying the ``cursor_factory`` to be ``ClientCursor`` when creating
the engine::

    from psycopg import ClientCursor

    client_side_engine = create_engine(
        "postgresql+psycopg://...",
        connect_args={"cursor_factory": ClientCursor},
    )

Similarly when using an async engine the ``AsyncClientCursor`` can be
specified::

    from psycopg import AsyncClientCursor

    client_side_engine = create_async_engine(
        "postgresql+psycopg://...",
        connect_args={"cursor_factory": AsyncClientCursor},
    )

.. seealso::

    `Client-side-binding cursors <https://www.psycopg.org/psycopg3/docs/advanced/cursors.html#client-side-binding-cursors>`_

"""  # noqa
from __future__ import annotations

from collections import deque
import logging
import re
from typing import cast
from typing import TYPE_CHECKING

from . import ranges
from ._psycopg_common import _PGDialect_common_psycopg
from ._psycopg_common import _PGExecutionContext_common_psycopg
from .base import INTERVAL
from .base import PGCompiler
from .base import PGIdentifierPreparer
from .base import REGCONFIG
from .json import JSON
from .json import JSONB
from .json import JSONPathType
from .types import CITEXT
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...sql import sqltypes
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only

if TYPE_CHECKING:
    from typing import Iterable

    from psycopg import AsyncConnection

logger = logging.getLogger("sqlalchemy.dialects.postgresql")


class _PGString(sqltypes.String):
    render_bind_cast = True


class _PGREGCONFIG(REGCONFIG):
    render_bind_cast = True


class _PGJSON(JSON):
    def bind_processor(self, dialect):
        return self._make_bind_processor(None, dialect._psycopg_Json)

    def result_processor(self, dialect, coltype):
        return None


class _PGJSONB(JSONB):
    def bind_processor(self, dialect):
        return self._make_bind_processor(None, dialect._psycopg_Jsonb)

    def result_processor(self, dialect, coltype):
        return None


class _PGJSONIntIndexType(sqltypes.JSON.JSONIntIndexType):
    __visit_name__ = "json_int_index"

    render_bind_cast = True


class _PGJSONStrIndexType(sqltypes.JSON.JSONStrIndexType):
    __visit_name__ = "json_str_index"

    render_bind_cast = True


class _PGJSONPathType(JSONPathType):
    pass


class _PGInterval(INTERVAL):
    render_bind_cast = True


class _PGTimeStamp(sqltypes.DateTime):
    render_bind_cast = True


class _PGDate(sqltypes.Date):
    render_bind_cast = True


class _PGTime(sqltypes.Time):
    render_bind_cast = True


class _PGInteger(sqltypes.Integer):
    render_bind_cast = True


class _PGSmallInteger(sqltypes.SmallInteger):
    render_bind_cast = True


class _PGNullType(sqltypes.NullType):
    render_bind_cast = True


class _PGBigInteger(sqltypes.BigInteger):
    render_bind_cast = True


class _PGBoolean(sqltypes.Boolean):
    render_bind_cast = True


class _PsycopgRange(ranges.AbstractSingleRangeImpl):
    def bind_processor(self, dialect):
        psycopg_Range = cast(PGDialect_psycopg, dialect)._psycopg_Range

        def to_range(value):
            if isinstance(value, ranges.Range):
                value = psycopg_Range(
                    value.lower, value.upper, value.bounds, value.empty
                )
            return value

        return to_range

    def result_processor(self, dialect, coltype):
        def to_range(value):
            if value is not None:
                value = ranges.Range(
                    value._lower,
                    value._upper,
                    bounds=value._bounds if value._bounds else "[)",
                    empty=not value._bounds,
                )
            return value

        return to_range


class _PsycopgMultiRange(ranges.AbstractMultiRangeImpl):
    def bind_processor(self, dialect):
        psycopg_Range = cast(PGDialect_psycopg, dialect)._psycopg_Range
        psycopg_Multirange = cast(
            PGDialect_psycopg, dialect
        )._psycopg_Multirange

        NoneType = type(None)

        def to_range(value):
            if isinstance(value, (str, NoneType, psycopg_Multirange)):
                return value

            return psycopg_Multirange(
                [
                    psycopg_Range(
                        element.lower,
                        element.upper,
                        element.bounds,
                        element.empty,
                    )
                    for element in cast("Iterable[ranges.Range]", value)
                ]
            )

        return to_range

    def result_processor(self, dialect, coltype):
        def to_range(value):
            if value is None:
                return None
            else:
                return ranges.MultiRange(
                    ranges.Range(
                        elem._lower,
                        elem._upper,
                        bounds=elem._bounds if elem._bounds else "[)",
                        empty=not elem._bounds,
                    )
                    for elem in value
                )

        return to_range


class PGExecutionContext_psycopg(_PGExecutionContext_common_psycopg):
    pass


class PGCompiler_psycopg(PGCompiler):
    pass


class PGIdentifierPreparer_psycopg(PGIdentifierPreparer):
    pass


def _log_notices(diagnostic):
    logger.info("%s: %s", diagnostic.severity, diagnostic.message_primary)


class PGDialect_psycopg(_PGDialect_common_psycopg):
    driver = "psycopg"

    supports_statement_cache = True
    supports_server_side_cursors = True
    default_paramstyle = "pyformat"
    supports_sane_multi_rowcount = True

    execution_ctx_cls = PGExecutionContext_psycopg
    statement_compiler = PGCompiler_psycopg
    preparer = PGIdentifierPreparer_psycopg
    psycopg_version = (0, 0)

    _has_native_hstore = True
    _psycopg_adapters_map = None

    colspecs = util.update_copy(
        _PGDialect_common_psycopg.colspecs,
        {
            sqltypes.String: _PGString,
            REGCONFIG: _PGREGCONFIG,
            JSON: _PGJSON,
            CITEXT: CITEXT,
            sqltypes.JSON: _PGJSON,
            JSONB: _PGJSONB,
            sqltypes.JSON.JSONPathType: _PGJSONPathType,
            sqltypes.JSON.JSONIntIndexType: _PGJSONIntIndexType,
            sqltypes.JSON.JSONStrIndexType: _PGJSONStrIndexType,
            sqltypes.Interval: _PGInterval,
            INTERVAL: _PGInterval,
            sqltypes.Date: _PGDate,
            sqltypes.DateTime: _PGTimeStamp,
            sqltypes.Time: _PGTime,
            sqltypes.Integer: _PGInteger,
            sqltypes.SmallInteger: _PGSmallInteger,
            sqltypes.BigInteger: _PGBigInteger,
            ranges.AbstractSingleRange: _PsycopgRange,
            ranges.AbstractMultiRange: _PsycopgMultiRange,
        },
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.dbapi:
            m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", self.dbapi.__version__)
            if m:
                self.psycopg_version = tuple(
                    int(x) for x in m.group(1, 2, 3) if x is not None
                )

            if self.psycopg_version < (3, 0, 2):
                raise ImportError(
                    "psycopg version 3.0.2 or higher is required."
                )

            from psycopg.adapt import AdaptersMap

            self._psycopg_adapters_map = adapters_map = AdaptersMap(
                self.dbapi.adapters
            )

            if self._native_inet_types is False:
                import psycopg.types.string

                adapters_map.register_loader(
                    "inet", psycopg.types.string.TextLoader
                )
                adapters_map.register_loader(
                    "cidr", psycopg.types.string.TextLoader
                )

            if self._json_deserializer:
                from psycopg.types.json import set_json_loads

                set_json_loads(self._json_deserializer, adapters_map)

            if self._json_serializer:
                from psycopg.types.json import set_json_dumps

                set_json_dumps(self._json_serializer, adapters_map)

    def create_connect_args(self, url):
        # see https://github.com/psycopg/psycopg/issues/83
        cargs, cparams = super().create_connect_args(url)

        if self._psycopg_adapters_map:
            cparams["context"] = self._psycopg_adapters_map
        if self.client_encoding is not None:
            cparams["client_encoding"] = self.client_encoding
        return cargs, cparams

    def _type_info_fetch(self, connection, name):
        from psycopg.types import TypeInfo

        return TypeInfo.fetch(connection.connection.driver_connection, name)

    def initialize(self, connection):
        super().initialize(connection)

        # PGDialect.initialize() checks server version for <= 8.2 and sets
        # this flag to False if so
        if not self.insert_returning:
            self.insert_executemany_returning = False

        # HSTORE can't be registered until we have a connection so that
        # we can look up its OID, so we set up this adapter in
        # initialize()
        if self.use_native_hstore:
            info = self._type_info_fetch(connection, "hstore")
            self._has_native_hstore = info is not None
            if self._has_native_hstore:
                from psycopg.types.hstore import register_hstore

                # register the adapter for connections made subsequent to
                # this one
                assert self._psycopg_adapters_map
                register_hstore(info, self._psycopg_adapters_map)

                # register the adapter for this connection
                assert connection.connection
                register_hstore(info, connection.connection.driver_connection)

    @classmethod
    def import_dbapi(cls):
        import psycopg

        return psycopg

    @classmethod
    def get_async_dialect_cls(cls, url):
        return PGDialectAsync_psycopg

    @util.memoized_property
    def _isolation_lookup(self):
        return {
            "READ COMMITTED": self.dbapi.IsolationLevel.READ_COMMITTED,
            "READ UNCOMMITTED": self.dbapi.IsolationLevel.READ_UNCOMMITTED,
            "REPEATABLE READ": self.dbapi.IsolationLevel.REPEATABLE_READ,
            "SERIALIZABLE": self.dbapi.IsolationLevel.SERIALIZABLE,
        }

    @util.memoized_property
    def _psycopg_Json(self):
        from psycopg.types import json

        return json.Json

    @util.memoized_property
    def _psycopg_Jsonb(self):
        from psycopg.types import json

        return json.Jsonb

    @util.memoized_property
    def _psycopg_TransactionStatus(self):
        from psycopg.pq import TransactionStatus

        return TransactionStatus

    @util.memoized_property
    def _psycopg_Range(self):
        from psycopg.types.range import Range

        return Range

    @util.memoized_property
    def _psycopg_Multirange(self):
        from psycopg.types.multirange import Multirange

        return Multirange

    def _do_isolation_level(self, connection, autocommit, isolation_level):
        connection.autocommit = autocommit
        connection.isolation_level = isolation_level

    def get_isolation_level(self, dbapi_connection):
        status_before = dbapi_connection.info.transaction_status
        value = super().get_isolation_level(dbapi_connection)

        # don't rely on psycopg providing enum symbols, compare with
        # eq/ne
        if status_before == self._psycopg_TransactionStatus.IDLE:
            dbapi_connection.rollback()
        return value

    def set_isolation_level(self, dbapi_connection, level):
        if level == "AUTOCOMMIT":
            self._do_isolation_level(
                dbapi_connection, autocommit=True, isolation_level=None
            )
        else:
            self._do_isolation_level(
                dbapi_connection,
                autocommit=False,
                isolation_level=self._isolation_lookup[level],
            )

    def set_readonly(self, connection, value):
        connection.read_only = value

    def get_readonly(self, connection):
        return connection.read_only

    def on_connect(self):
        def notices(conn):
            conn.add_notice_handler(_log_notices)

        fns = [notices]

        if self.isolation_level is not None:

            def on_connect(conn):
                self.set_isolation_level(conn, self.isolation_level)

            fns.append(on_connect)

        # fns always has the notices function
        def on_connect(conn):
            for fn in fns:
                fn(conn)

        return on_connect

    def is_disconnect(self, e, connection, cursor):
        if isinstance(e, self.dbapi.Error) and connection is not None:
            if connection.closed or connection.broken:
                return True
        return False

    def _do_prepared_twophase(self, connection, command, recover=False):
        dbapi_conn = connection.connection.dbapi_connection
        if (
            recover
            # don't rely on psycopg providing enum symbols, compare with
            # eq/ne
            or dbapi_conn.info.transaction_status
            != self._psycopg_TransactionStatus.IDLE
        ):
            dbapi_conn.rollback()
        before_autocommit = dbapi_conn.autocommit
        try:
            if not before_autocommit:
                self._do_autocommit(dbapi_conn, True)
            dbapi_conn.execute(command)
        finally:
            if not before_autocommit:
                self._do_autocommit(dbapi_conn, before_autocommit)

    def do_rollback_twophase(
        self, connection, xid, is_prepared=True, recover=False
    ):
        if is_prepared:
            self._do_prepared_twophase(
                connection, f"ROLLBACK PREPARED '{xid}'", recover=recover
            )
        else:
            self.do_rollback(connection.connection)

    def do_commit_twophase(
        self, connection, xid, is_prepared=True, recover=False
    ):
        if is_prepared:
            self._do_prepared_twophase(
                connection, f"COMMIT PREPARED '{xid}'", recover=recover
            )
        else:
            self.do_commit(connection.connection)

    @util.memoized_property
    def _dialect_specific_select_one(self):
        return ";"


class AsyncAdapt_psycopg_cursor:
    __slots__ = ("_cursor", "await_", "_rows")

    _psycopg_ExecStatus = None

    def __init__(self, cursor, await_) -> None:
        self._cursor = cursor
        self.await_ = await_
        self._rows = deque()

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    @property
    def arraysize(self):
        return self._cursor.arraysize

    @arraysize.setter
    def arraysize(self, value):
        self._cursor.arraysize = value

    def close(self):
        self._rows.clear()
        # Normal cursor just call _close() in a non-sync way.
        self._cursor._close()

    def execute(self, query, params=None, **kw):
        result = self.await_(self._cursor.execute(query, params, **kw))
        # sqlalchemy result is not async, so need to pull all rows here
        res = self._cursor.pgresult

        # don't rely on psycopg providing enum symbols, compare with
        # eq/ne
        if res and res.status == self._psycopg_ExecStatus.TUPLES_OK:
            rows = self.await_(self._cursor.fetchall())
            self._rows = deque(rows)
        return result

    def executemany(self, query, params_seq):
        return self.await_(self._cursor.executemany(query, params_seq))

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
            size = self._cursor.arraysize

        rr = self._rows
        return [rr.popleft() for _ in range(min(size, len(rr)))]

    def fetchall(self):
        retval = list(self._rows)
        self._rows.clear()
        return retval


class AsyncAdapt_psycopg_ss_cursor(AsyncAdapt_psycopg_cursor):
    def execute(self, query, params=None, **kw):
        self.await_(self._cursor.execute(query, params, **kw))
        return self

    def close(self):
        self.await_(self._cursor.close())

    def fetchone(self):
        return self.await_(self._cursor.fetchone())

    def fetchmany(self, size=0):
        return self.await_(self._cursor.fetchmany(size))

    def fetchall(self):
        return self.await_(self._cursor.fetchall())

    def __iter__(self):
        iterator = self._cursor.__aiter__()
        while True:
            try:
                yield self.await_(iterator.__anext__())
            except StopAsyncIteration:
                break


class AsyncAdapt_psycopg_connection(AdaptedConnection):
    _connection: AsyncConnection
    __slots__ = ()
    await_ = staticmethod(await_only)

    def __init__(self, connection) -> None:
        self._connection = connection

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def execute(self, query, params=None, **kw):
        cursor = self.await_(self._connection.execute(query, params, **kw))
        return AsyncAdapt_psycopg_cursor(cursor, self.await_)

    def cursor(self, *args, **kw):
        cursor = self._connection.cursor(*args, **kw)
        if hasattr(cursor, "name"):
            return AsyncAdapt_psycopg_ss_cursor(cursor, self.await_)
        else:
            return AsyncAdapt_psycopg_cursor(cursor, self.await_)

    def commit(self):
        self.await_(self._connection.commit())

    def rollback(self):
        self.await_(self._connection.rollback())

    def close(self):
        self.await_(self._connection.close())

    @property
    def autocommit(self):
        return self._connection.autocommit

    @autocommit.setter
    def autocommit(self, value):
        self.set_autocommit(value)

    def set_autocommit(self, value):
        self.await_(self._connection.set_autocommit(value))

    def set_isolation_level(self, value):
        self.await_(self._connection.set_isolation_level(value))

    def set_read_only(self, value):
        self.await_(self._connection.set_read_only(value))

    def set_deferrable(self, value):
        self.await_(self._connection.set_deferrable(value))


class AsyncAdaptFallback_psycopg_connection(AsyncAdapt_psycopg_connection):
    __slots__ = ()
    await_ = staticmethod(await_fallback)


class PsycopgAdaptDBAPI:
    def __init__(self, psycopg) -> None:
        self.psycopg = psycopg

        for k, v in self.psycopg.__dict__.items():
            if k != "connect":
                self.__dict__[k] = v

    def connect(self, *arg, **kw):
        async_fallback = kw.pop("async_fallback", False)
        creator_fn = kw.pop(
            "async_creator_fn", self.psycopg.AsyncConnection.connect
        )
        if util.asbool(async_fallback):
            return AsyncAdaptFallback_psycopg_connection(
                await_fallback(creator_fn(*arg, **kw))
            )
        else:
            return AsyncAdapt_psycopg_connection(
                await_only(creator_fn(*arg, **kw))
            )


class PGDialectAsync_psycopg(PGDialect_psycopg):
    is_async = True
    supports_statement_cache = True

    @classmethod
    def import_dbapi(cls):
        import psycopg
        from psycopg.pq import ExecStatus

        AsyncAdapt_psycopg_cursor._psycopg_ExecStatus = ExecStatus

        return PsycopgAdaptDBAPI(psycopg)

    @classmethod
    def get_pool_class(cls, url):
        async_fallback = url.query.get("async_fallback", False)

        if util.asbool(async_fallback):
            return pool.FallbackAsyncAdaptedQueuePool
        else:
            return pool.AsyncAdaptedQueuePool

    def _type_info_fetch(self, connection, name):
        from psycopg.types import TypeInfo

        adapted = connection.connection
        return adapted.await_(TypeInfo.fetch(adapted.driver_connection, name))

    def _do_isolation_level(self, connection, autocommit, isolation_level):
        connection.set_autocommit(autocommit)
        connection.set_isolation_level(isolation_level)

    def _do_autocommit(self, connection, value):
        connection.set_autocommit(value)

    def set_readonly(self, connection, value):
        connection.set_read_only(value)

    def set_deferrable(self, connection, value):
        connection.set_deferrable(value)

    def get_driver_connection(self, connection):
        return connection._connection


dialect = PGDialect_psycopg
dialect_async = PGDialectAsync_psycopg

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\heurisch.py ===
from __future__ import annotations

from collections import defaultdict
from functools import reduce
from itertools import permutations

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.symbol import Wild, Dummy, Symbol
from sympy.core.basic import sympify
from sympy.core.numbers import Rational, pi, I
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.traversal import iterfreeargs

from sympy.functions import exp, sin, cos, tan, cot, asin, atan
from sympy.functions import log, sinh, cosh, tanh, coth, asinh
from sympy.functions import sqrt, erf, erfi, li, Ei
from sympy.functions import besselj, bessely, besseli, besselk
from sympy.functions import hankel1, hankel2, jn, yn
from sympy.functions.elementary.complexes import Abs, re, im, sign, arg
from sympy.functions.elementary.exponential import LambertW
from sympy.functions.elementary.integers import floor, ceiling
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import Heaviside, DiracDelta

from sympy.simplify.radsimp import collect

from sympy.logic.boolalg import And, Or
from sympy.utilities.iterables import uniq

from sympy.polys import quo, gcd, lcm, factor_list, cancel, PolynomialError
from sympy.polys.monomials import itermonomials
from sympy.polys.polyroots import root_factors

from sympy.polys.rings import PolyRing
from sympy.polys.solvers import solve_lin_sys
from sympy.polys.constructor import construct_domain

from sympy.integrals.integrals import integrate


def components(f, x):
    """
    Returns a set of all functional components of the given expression
    which includes symbols, function applications and compositions and
    non-integer powers. Fractional powers are collected with
    minimal, positive exponents.

    Examples
    ========

    >>> from sympy import cos, sin
    >>> from sympy.abc import x
    >>> from sympy.integrals.heurisch import components

    >>> components(sin(x)*cos(x)**2, x)
    {x, sin(x), cos(x)}

    See Also
    ========

    heurisch
    """
    result = set()

    if f.has_free(x):
        if f.is_symbol and f.is_commutative:
            result.add(f)
        elif f.is_Function or f.is_Derivative:
            for g in f.args:
                result |= components(g, x)

            result.add(f)
        elif f.is_Pow:
            result |= components(f.base, x)

            if not f.exp.is_Integer:
                if f.exp.is_Rational:
                    result.add(f.base**Rational(1, f.exp.q))
                else:
                    result |= components(f.exp, x) | {f}
        else:
            for g in f.args:
                result |= components(g, x)

    return result

# name -> [] of symbols
_symbols_cache: dict[str, list[Dummy]] = {}


# NB @cacheit is not convenient here
def _symbols(name, n):
    """get vector of symbols local to this module"""
    try:
        lsyms = _symbols_cache[name]
    except KeyError:
        lsyms = []
        _symbols_cache[name] = lsyms

    while len(lsyms) < n:
        lsyms.append( Dummy('%s%i' % (name, len(lsyms))) )

    return lsyms[:n]


def heurisch_wrapper(f, x, rewrite=False, hints=None, mappings=None, retries=3,
                     degree_offset=0, unnecessary_permutations=None,
                     _try_heurisch=None):
    """
    A wrapper around the heurisch integration algorithm.

    Explanation
    ===========

    This method takes the result from heurisch and checks for poles in the
    denominator. For each of these poles, the integral is reevaluated, and
    the final integration result is given in terms of a Piecewise.

    Examples
    ========

    >>> from sympy import cos, symbols
    >>> from sympy.integrals.heurisch import heurisch, heurisch_wrapper
    >>> n, x = symbols('n x')
    >>> heurisch(cos(n*x), x)
    sin(n*x)/n
    >>> heurisch_wrapper(cos(n*x), x)
    Piecewise((sin(n*x)/n, Ne(n, 0)), (x, True))

    See Also
    ========

    heurisch
    """
    from sympy.solvers.solvers import solve, denoms
    f = sympify(f)
    if not f.has_free(x):
        return f*x

    res = heurisch(f, x, rewrite, hints, mappings, retries, degree_offset,
                   unnecessary_permutations, _try_heurisch)
    if not isinstance(res, Basic):
        return res

    # We consider each denominator in the expression, and try to find
    # cases where one or more symbolic denominator might be zero. The
    # conditions for these cases are stored in the list slns.
    #
    # Since denoms returns a set we use ordered. This is important because the
    # ordering of slns determines the order of the resulting Piecewise so we
    # need a deterministic order here to make the output deterministic.
    slns = []
    for d in ordered(denoms(res)):
        try:
            slns += solve([d], dict=True, exclude=(x,))
        except NotImplementedError:
            pass
    if not slns:
        return res
    slns = list(uniq(slns))
    # Remove the solutions corresponding to poles in the original expression.
    slns0 = []
    for d in denoms(f):
        try:
            slns0 += solve([d], dict=True, exclude=(x,))
        except NotImplementedError:
            pass
    slns = [s for s in slns if s not in slns0]
    if not slns:
        return res
    if len(slns) > 1:
        eqs = []
        for sub_dict in slns:
            eqs.extend([Eq(key, value) for key, value in sub_dict.items()])
        slns = solve(eqs, dict=True, exclude=(x,)) + slns
    # For each case listed in the list slns, we reevaluate the integral.
    pairs = []
    for sub_dict in slns:
        expr = heurisch(f.subs(sub_dict), x, rewrite, hints, mappings, retries,
                        degree_offset, unnecessary_permutations,
                        _try_heurisch)
        cond = And(*[Eq(key, value) for key, value in sub_dict.items()])
        generic = Or(*[Ne(key, value) for key, value in sub_dict.items()])
        if expr is None:
            expr = integrate(f.subs(sub_dict),x)
        pairs.append((expr, cond))
    # If there is one condition, put the generic case first. Otherwise,
    # doing so may lead to longer Piecewise formulas
    if len(pairs) == 1:
        pairs = [(heurisch(f, x, rewrite, hints, mappings, retries,
                              degree_offset, unnecessary_permutations,
                              _try_heurisch),
                              generic),
                 (pairs[0][0], True)]
    else:
        pairs.append((heurisch(f, x, rewrite, hints, mappings, retries,
                              degree_offset, unnecessary_permutations,
                              _try_heurisch),
                              True))
    return Piecewise(*pairs)

class BesselTable:
    """
    Derivatives of Bessel functions of orders n and n-1
    in terms of each other.

    See the docstring of DiffCache.
    """

    def __init__(self):
        self.table = {}
        self.n = Dummy('n')
        self.z = Dummy('z')
        self._create_table()

    def _create_table(t):
        table, n, z = t.table, t.n, t.z
        for f in (besselj, bessely, hankel1, hankel2):
            table[f] = (f(n-1, z) - n*f(n, z)/z,
                        (n-1)*f(n-1, z)/z - f(n, z))

        f = besseli
        table[f] = (f(n-1, z) - n*f(n, z)/z,
                    (n-1)*f(n-1, z)/z + f(n, z))
        f = besselk
        table[f] = (-f(n-1, z) - n*f(n, z)/z,
                    (n-1)*f(n-1, z)/z - f(n, z))

        for f in (jn, yn):
            table[f] = (f(n-1, z) - (n+1)*f(n, z)/z,
                        (n-1)*f(n-1, z)/z - f(n, z))

    def diffs(t, f, n, z):
        if f in t.table:
            diff0, diff1 = t.table[f]
            repl = [(t.n, n), (t.z, z)]
            return (diff0.subs(repl), diff1.subs(repl))

    def has(t, f):
        return f in t.table

_bessel_table = None

class DiffCache:
    """
    Store for derivatives of expressions.

    Explanation
    ===========

    The standard form of the derivative of a Bessel function of order n
    contains two Bessel functions of orders n-1 and n+1, respectively.
    Such forms cannot be used in parallel Risch algorithm, because
    there is a linear recurrence relation between the three functions
    while the algorithm expects that functions and derivatives are
    represented in terms of algebraically independent transcendentals.

    The solution is to take two of the functions, e.g., those of orders
    n and n-1, and to express the derivatives in terms of the pair.
    To guarantee that the proper form is used the two derivatives are
    cached as soon as one is encountered.

    Derivatives of other functions are also cached at no extra cost.
    All derivatives are with respect to the same variable `x`.
    """

    def __init__(self, x):
        self.cache = {}
        self.x = x

        global _bessel_table
        if not _bessel_table:
            _bessel_table = BesselTable()

    def get_diff(self, f):
        cache = self.cache

        if f in cache:
            pass
        elif (not hasattr(f, 'func') or
            not _bessel_table.has(f.func)):
            cache[f] = cancel(f.diff(self.x))
        else:
            n, z = f.args
            d0, d1 = _bessel_table.diffs(f.func, n, z)
            dz = self.get_diff(z)
            cache[f] = d0*dz
            cache[f.func(n-1, z)] = d1*dz

        return cache[f]

def heurisch(f, x, rewrite=False, hints=None, mappings=None, retries=3,
             degree_offset=0, unnecessary_permutations=None,
             _try_heurisch=None):
    """
    Compute indefinite integral using heuristic Risch algorithm.

    Explanation
    ===========

    This is a heuristic approach to indefinite integration in finite
    terms using the extended heuristic (parallel) Risch algorithm, based
    on Manuel Bronstein's "Poor Man's Integrator".

    The algorithm supports various classes of functions including
    transcendental elementary or special functions like Airy,
    Bessel, Whittaker and Lambert.

    Note that this algorithm is not a decision procedure. If it isn't
    able to compute the antiderivative for a given function, then this is
    not a proof that such a functions does not exist.  One should use
    recursive Risch algorithm in such case.  It's an open question if
    this algorithm can be made a full decision procedure.

    This is an internal integrator procedure. You should use top level
    'integrate' function in most cases, as this procedure needs some
    preprocessing steps and otherwise may fail.

    Specification
    =============

     heurisch(f, x, rewrite=False, hints=None)

       where
         f : expression
         x : symbol

         rewrite -> force rewrite 'f' in terms of 'tan' and 'tanh'
         hints   -> a list of functions that may appear in anti-derivate

          - hints = None          --> no suggestions at all
          - hints = [ ]           --> try to figure out
          - hints = [f1, ..., fn] --> we know better

    Examples
    ========

    >>> from sympy import tan
    >>> from sympy.integrals.heurisch import heurisch
    >>> from sympy.abc import x, y

    >>> heurisch(y*tan(x), x)
    y*log(tan(x)**2 + 1)/2

    See Manuel Bronstein's "Poor Man's Integrator":

    References
    ==========

    .. [1] https://www-sop.inria.fr/cafe/Manuel.Bronstein/pmint/index.html

    For more information on the implemented algorithm refer to:

    .. [2] K. Geddes, L. Stefanus, On the Risch-Norman Integration
       Method and its Implementation in Maple, Proceedings of
       ISSAC'89, ACM Press, 212-217.

    .. [3] J. H. Davenport, On the Parallel Risch Algorithm (I),
       Proceedings of EUROCAM'82, LNCS 144, Springer, 144-157.

    .. [4] J. H. Davenport, On the Parallel Risch Algorithm (III):
       Use of Tangents, SIGSAM Bulletin 16 (1982), 3-6.

    .. [5] J. H. Davenport, B. M. Trager, On the Parallel Risch
       Algorithm (II), ACM Transactions on Mathematical
       Software 11 (1985), 356-362.

    See Also
    ========

    sympy.integrals.integrals.Integral.doit
    sympy.integrals.integrals.Integral
    sympy.integrals.heurisch.components
    """
    f = sympify(f)

    # There are some functions that Heurisch cannot currently handle,
    # so do not even try.
    # Set _try_heurisch=True to skip this check
    if _try_heurisch is not True:
        if f.has(Abs, re, im, sign, Heaviside, DiracDelta, floor, ceiling, arg):
            return

    if not f.has_free(x):
        return f*x

    if not f.is_Add:
        indep, f = f.as_independent(x)
    else:
        indep = S.One

    rewritables = {
        (sin, cos, cot): tan,
        (sinh, cosh, coth): tanh,
    }

    if rewrite:
        for candidates, rule in rewritables.items():
            f = f.rewrite(candidates, rule)
    else:
        for candidates in rewritables.keys():
            if f.has(*candidates):
                break
        else:
            rewrite = True

    terms = components(f, x)
    dcache = DiffCache(x)

    if hints is not None:
        if not hints:
            a = Wild('a', exclude=[x])
            b = Wild('b', exclude=[x])
            c = Wild('c', exclude=[x])

            for g in set(terms):  # using copy of terms
                if g.is_Function:
                    if isinstance(g, li):
                        M = g.args[0].match(a*x**b)

                        if M is not None:
                            terms.add( x*(li(M[a]*x**M[b]) - (M[a]*x**M[b])**(-1/M[b])*Ei((M[b]+1)*log(M[a]*x**M[b])/M[b])) )
                            #terms.add( x*(li(M[a]*x**M[b]) - (x**M[b])**(-1/M[b])*Ei((M[b]+1)*log(M[a]*x**M[b])/M[b])) )
                            #terms.add( x*(li(M[a]*x**M[b]) - x*Ei((M[b]+1)*log(M[a]*x**M[b])/M[b])) )
                            #terms.add( li(M[a]*x**M[b]) - Ei((M[b]+1)*log(M[a]*x**M[b])/M[b]) )

                    elif isinstance(g, exp):
                        M = g.args[0].match(a*x**2)

                        if M is not None:
                            if M[a].is_positive:
                                terms.add(erfi(sqrt(M[a])*x))
                            else: # M[a].is_negative or unknown
                                terms.add(erf(sqrt(-M[a])*x))

                        M = g.args[0].match(a*x**2 + b*x + c)

                        if M is not None:
                            if M[a].is_positive:
                                terms.add(sqrt(pi/4*(-M[a]))*exp(M[c] - M[b]**2/(4*M[a]))*
                                          erfi(sqrt(M[a])*x + M[b]/(2*sqrt(M[a]))))
                            elif M[a].is_negative:
                                terms.add(sqrt(pi/4*(-M[a]))*exp(M[c] - M[b]**2/(4*M[a]))*
                                          erf(sqrt(-M[a])*x - M[b]/(2*sqrt(-M[a]))))

                        M = g.args[0].match(a*log(x)**2)

                        if M is not None:
                            if M[a].is_positive:
                                terms.add(erfi(sqrt(M[a])*log(x) + 1/(2*sqrt(M[a]))))
                            if M[a].is_negative:
                                terms.add(erf(sqrt(-M[a])*log(x) - 1/(2*sqrt(-M[a]))))

                elif g.is_Pow:
                    if g.exp.is_Rational and g.exp.q == 2:
                        M = g.base.match(a*x**2 + b)

                        if M is not None and M[b].is_positive:
                            if M[a].is_positive:
                                terms.add(asinh(sqrt(M[a]/M[b])*x))
                            elif M[a].is_negative:
                                terms.add(asin(sqrt(-M[a]/M[b])*x))

                        M = g.base.match(a*x**2 - b)

                        if M is not None and M[b].is_positive:
                            if M[a].is_positive:
                                dF = 1/sqrt(M[a]*x**2 - M[b])
                                F = log(2*sqrt(M[a])*sqrt(M[a]*x**2 - M[b]) + 2*M[a]*x)/sqrt(M[a])
                                dcache.cache[F] = dF  # hack: F.diff(x) doesn't automatically simplify to f
                                terms.add(F)
                            elif M[a].is_negative:
                                terms.add(-M[b]/2*sqrt(-M[a])*
                                           atan(sqrt(-M[a])*x/sqrt(M[a]*x**2 - M[b])))

        else:
            terms |= set(hints)

    for g in set(terms):  # using copy of terms
        terms |= components(dcache.get_diff(g), x)

    # XXX: The commented line below makes heurisch more deterministic wrt
    # PYTHONHASHSEED and the iteration order of sets. There are other places
    # where sets are iterated over but this one is possibly the most important.
    # Theoretically the order here should not matter but different orderings
    # can expose potential bugs in the different code paths so potentially it
    # is better to keep the non-determinism.
    #
    # terms = list(ordered(terms))

    # TODO: caching is significant factor for why permutations work at all. Change this.
    V = _symbols('x', len(terms))


    # sort mapping expressions from largest to smallest (last is always x).
    mapping = list(reversed(list(zip(*ordered(                          #
        [(a[0].as_independent(x)[1], a) for a in zip(terms, V)])))[1])) #
    rev_mapping = {v: k for k, v in mapping}                            #
    if mappings is None:                                                #
        # optimizing the number of permutations of mapping              #
        assert mapping[-1][0] == x # if not, find it and correct this comment
        unnecessary_permutations = [mapping.pop(-1)]
        # permute types of objects
        types = defaultdict(list)
        for i in mapping:
            e, _ = i
            types[type(e)].append(i)
        mapping = [types[i] for i in types]
        def _iter_mappings():
            for i in permutations(mapping):
                # make the expression of a given type be ordered
                yield [j for i in i for j in ordered(i)]
        mappings = _iter_mappings()
    else:
        unnecessary_permutations = unnecessary_permutations or []

    def _substitute(expr):
        return expr.subs(mapping)

    for mapping in mappings:
        mapping = list(mapping)
        mapping = mapping + unnecessary_permutations
        diffs = [ _substitute(dcache.get_diff(g)) for g in terms ]
        denoms = [ g.as_numer_denom()[1] for g in diffs ]
        if all(h.is_polynomial(*V) for h in denoms) and _substitute(f).is_rational_function(*V):
            denom = reduce(lambda p, q: lcm(p, q, *V), denoms)
            break
    else:
        if not rewrite:
            result = heurisch(f, x, rewrite=True, hints=hints,
                unnecessary_permutations=unnecessary_permutations)

            if result is not None:
                return indep*result
        return None

    numers = [ cancel(denom*g) for g in diffs ]
    def _derivation(h):
        return Add(*[ d * h.diff(v) for d, v in zip(numers, V) ])

    def _deflation(p):
        for y in V:
            if not p.has(y):
                continue

            if _derivation(p) is not S.Zero:
                c, q = p.as_poly(y).primitive()
                return _deflation(c)*gcd(q, q.diff(y)).as_expr()

        return p

    def _splitter(p):
        for y in V:
            if not p.has(y):
                continue

            if _derivation(y) is not S.Zero:
                c, q = p.as_poly(y).primitive()

                q = q.as_expr()

                h = gcd(q, _derivation(q), y)
                s = quo(h, gcd(q, q.diff(y), y), y)

                c_split = _splitter(c)

                if s.as_poly(y).degree() == 0:
                    return (c_split[0], q * c_split[1])

                q_split = _splitter(cancel(q / s))

                return (c_split[0]*q_split[0]*s, c_split[1]*q_split[1])

        return (S.One, p)

    special = {}

    for term in terms:
        if term.is_Function:
            if isinstance(term, tan):
                special[1 + _substitute(term)**2] = False
            elif isinstance(term, tanh):
                special[1 + _substitute(term)] = False
                special[1 - _substitute(term)] = False
            elif isinstance(term, LambertW):
                special[_substitute(term)] = True

    F = _substitute(f)

    P, Q = F.as_numer_denom()

    u_split = _splitter(denom)
    v_split = _splitter(Q)

    polys = set(list(v_split) + [ u_split[0] ] + list(special.keys()))

    s = u_split[0] * Mul(*[ k for k, v in special.items() if v ])
    polified = [ p.as_poly(*V) for p in [s, P, Q] ]

    if None in polified:
        return None

    #--- definitions for _integrate
    a, b, c = [ p.total_degree() for p in polified ]

    poly_denom = (s * v_split[0] * _deflation(v_split[1])).as_expr()

    def _exponent(g):
        if g.is_Pow:
            if g.exp.is_Rational and g.exp.q != 1:
                if g.exp.p > 0:
                    return g.exp.p + g.exp.q - 1
                else:
                    return abs(g.exp.p + g.exp.q)
            else:
                return 1
        elif not g.is_Atom and g.args:
            return max(_exponent(h) for h in g.args)
        else:
            return 1

    A, B = _exponent(f), a + max(b, c)

    if A > 1 and B > 1:
        monoms = tuple(ordered(itermonomials(V, A + B - 1 + degree_offset)))
    else:
        monoms = tuple(ordered(itermonomials(V, A + B + degree_offset)))

    poly_coeffs = _symbols('A', len(monoms))

    poly_part = Add(*[ poly_coeffs[i]*monomial
        for i, monomial in enumerate(monoms) ])

    reducibles = set()

    for poly in ordered(polys):
        coeff, factors = factor_list(poly, *V)
        reducibles.add(coeff)
        reducibles.update(fact for fact, mul in factors)

    def _integrate(field=None):
        atans = set()
        pairs = set()

        if field == 'Q':
            irreducibles = set(reducibles)
        else:
            setV = set(V)
            irreducibles = set()
            for poly in ordered(reducibles):
                zV = setV & set(iterfreeargs(poly))
                for z in ordered(zV):
                    s = set(root_factors(poly, z, filter=field))
                    irreducibles |= s
                    break

        log_part, atan_part = [], []

        for poly in ordered(irreducibles):
            m = collect(poly, I, evaluate=False)
            y = m.get(I, S.Zero)
            if y:
                x = m.get(S.One, S.Zero)
                if x.has(I) or y.has(I):
                    continue  # nontrivial x + I*y
                pairs.add((x, y))
                irreducibles.remove(poly)

        while pairs:
            x, y = pairs.pop()
            if (x, -y) in pairs:
                pairs.remove((x, -y))
                # Choosing b with no minus sign
                if y.could_extract_minus_sign():
                    y = -y
                irreducibles.add(x*x + y*y)
                atans.add(atan(x/y))
            else:
                irreducibles.add(x + I*y)


        B = _symbols('B', len(irreducibles))
        C = _symbols('C', len(atans))

        # Note: the ordering matters here
        for poly, b in reversed(list(zip(ordered(irreducibles), B))):
            if poly.has(*V):
                poly_coeffs.append(b)
                log_part.append(b * log(poly))

        for poly, c in reversed(list(zip(ordered(atans), C))):
            if poly.has(*V):
                poly_coeffs.append(c)
                atan_part.append(c * poly)

        # TODO: Currently it's better to use symbolic expressions here instead
        # of rational functions, because it's simpler and FracElement doesn't
        # give big speed improvement yet. This is because cancellation is slow
        # due to slow polynomial GCD algorithms. If this gets improved then
        # revise this code.
        candidate = poly_part/poly_denom + Add(*log_part) + Add(*atan_part)
        h = F - _derivation(candidate) / denom
        raw_numer = h.as_numer_denom()[0]

        # Rewrite raw_numer as a polynomial in K[coeffs][V] where K is a field
        # that we have to determine. We can't use simply atoms() because log(3),
        # sqrt(y) and similar expressions can appear, leading to non-trivial
        # domains.
        syms = set(poly_coeffs) | set(V)
        non_syms = set()

        def find_non_syms(expr):
            if expr.is_Integer or expr.is_Rational:
                pass # ignore trivial numbers
            elif expr in syms:
                pass # ignore variables
            elif not expr.has_free(*syms):
                non_syms.add(expr)
            elif expr.is_Add or expr.is_Mul or expr.is_Pow:
                list(map(find_non_syms, expr.args))
            else:
                # TODO: Non-polynomial expression. This should have been
                # filtered out at an earlier stage.
                raise PolynomialError

        try:
            find_non_syms(raw_numer)
        except PolynomialError:
            return None
        else:
            ground, _ = construct_domain(non_syms, field=True)

        coeff_ring = PolyRing(poly_coeffs, ground)
        ring = PolyRing(V, coeff_ring)
        try:
            numer = ring.from_expr(raw_numer)
        except ValueError:
            raise PolynomialError
        solution = solve_lin_sys(numer.coeffs(), coeff_ring, _raw=False)

        if solution is None:
            return None
        else:
            return candidate.xreplace(solution).xreplace(
                dict(zip(poly_coeffs, [S.Zero]*len(poly_coeffs))))

    if all(isinstance(_, Symbol) for _ in V):
        more_free = F.free_symbols - set(V)
    else:
        Fd = F.as_dummy()
        more_free = Fd.xreplace(dict(zip(V, (Dummy() for _ in V)))
            ).free_symbols & Fd.free_symbols
    if not more_free:
        # all free generators are identified in V
        solution = _integrate('Q')

        if solution is None:
            solution = _integrate()
    else:
        solution = _integrate()

    if solution is not None:
        antideriv = solution.subs(rev_mapping)
        antideriv = cancel(antideriv).expand()

        if antideriv.is_Add:
            antideriv = antideriv.as_independent(x)[1]

        return indep*antideriv
    else:
        if retries >= 0:
            result = heurisch(f, x, mappings=mappings, rewrite=rewrite, hints=hints, retries=retries - 1, unnecessary_permutations=unnecessary_permutations)

            if result is not None:
                return indep*result

        return None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\plugin\plugin_base.py ===
# testing/plugin/plugin_base.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


from __future__ import annotations

import abc
from argparse import Namespace
import configparser
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any

from sqlalchemy.testing import asyncio

"""Testing extensions.

this module is designed to work as a testing-framework-agnostic library,
created so that multiple test frameworks can be supported at once
(mostly so that we can migrate to new ones). The current target
is pytest.

"""

# flag which indicates we are in the SQLAlchemy testing suite,
# and not that of Alembic or a third party dialect.
bootstrapped_as_sqlalchemy = False

log = logging.getLogger("sqlalchemy.testing.plugin_base")

# late imports
fixtures = None
engines = None
exclusions = None
warnings = None
profiling = None
provision = None
assertions = None
requirements = None
config = None
testing = None
util = None
file_config = None

logging = None
include_tags = set()
exclude_tags = set()
options: Namespace = None  # type: ignore


def setup_options(make_option):
    make_option(
        "--log-info",
        action="callback",
        type=str,
        callback=_log,
        help="turn on info logging for <LOG> (multiple OK)",
    )
    make_option(
        "--log-debug",
        action="callback",
        type=str,
        callback=_log,
        help="turn on debug logging for <LOG> (multiple OK)",
    )
    make_option(
        "--db",
        action="append",
        type=str,
        dest="db",
        help="Use prefab database uri. Multiple OK, "
        "first one is run by default.",
    )
    make_option(
        "--dbs",
        action="callback",
        zeroarg_callback=_list_dbs,
        help="List available prefab dbs",
    )
    make_option(
        "--dburi",
        action="append",
        type=str,
        dest="dburi",
        help="Database uri.  Multiple OK, first one is run by default.",
    )
    make_option(
        "--dbdriver",
        action="append",
        type=str,
        dest="dbdriver",
        help="Additional database drivers to include in tests.  "
        "These are linked to the existing database URLs by the "
        "provisioning system.",
    )
    make_option(
        "--dropfirst",
        action="store_true",
        dest="dropfirst",
        help="Drop all tables in the target database first",
    )
    make_option(
        "--disable-asyncio",
        action="store_true",
        help="disable test / fixtures / provisoning running in asyncio",
    )
    make_option(
        "--backend-only",
        action="callback",
        zeroarg_callback=_set_tag_include("backend"),
        help=(
            "Run only tests marked with __backend__ or __sparse_backend__; "
            "this is now equivalent to the pytest -m backend mark expression"
        ),
    )
    make_option(
        "--nomemory",
        action="callback",
        zeroarg_callback=_set_tag_exclude("memory_intensive"),
        help="Don't run memory profiling tests; "
        "this is now equivalent to the pytest -m 'not memory_intensive' "
        "mark expression",
    )
    make_option(
        "--notimingintensive",
        action="callback",
        zeroarg_callback=_set_tag_exclude("timing_intensive"),
        help="Don't run timing intensive tests; "
        "this is now equivalent to the pytest -m 'not timing_intensive' "
        "mark expression",
    )
    make_option(
        "--nomypy",
        action="callback",
        zeroarg_callback=_set_tag_exclude("mypy"),
        help="Don't run mypy typing tests; "
        "this is now equivalent to the pytest -m 'not mypy' mark expression",
    )
    make_option(
        "--profile-sort",
        type=str,
        default="cumulative",
        dest="profilesort",
        help="Type of sort for profiling standard output",
    )
    make_option(
        "--profile-dump",
        type=str,
        dest="profiledump",
        help="Filename where a single profile run will be dumped",
    )
    make_option(
        "--low-connections",
        action="store_true",
        dest="low_connections",
        help="Use a low number of distinct connections - "
        "i.e. for Oracle TNS",
    )
    make_option(
        "--write-idents",
        type=str,
        dest="write_idents",
        help="write out generated follower idents to <file>, "
        "when -n<num> is used",
    )
    make_option(
        "--requirements",
        action="callback",
        type=str,
        callback=_requirements_opt,
        help="requirements class for testing, overrides setup.cfg",
    )
    make_option(
        "--include-tag",
        action="callback",
        callback=_include_tag,
        type=str,
        help="Include tests with tag <tag>; "
        "legacy, use pytest -m 'tag' instead",
    )
    make_option(
        "--exclude-tag",
        action="callback",
        callback=_exclude_tag,
        type=str,
        help="Exclude tests with tag <tag>; "
        "legacy, use pytest -m 'not tag' instead",
    )
    make_option(
        "--write-profiles",
        action="store_true",
        dest="write_profiles",
        default=False,
        help="Write/update failing profiling data.",
    )
    make_option(
        "--force-write-profiles",
        action="store_true",
        dest="force_write_profiles",
        default=False,
        help="Unconditionally write/update profiling data.",
    )
    make_option(
        "--dump-pyannotate",
        type=str,
        dest="dump_pyannotate",
        help="Run pyannotate and dump json info to given file",
    )
    make_option(
        "--mypy-extra-test-path",
        type=str,
        action="append",
        default=[],
        dest="mypy_extra_test_paths",
        help="Additional test directories to add to the mypy tests. "
        "This is used only when running mypy tests. Multiple OK",
    )
    # db specific options
    make_option(
        "--postgresql-templatedb",
        type=str,
        help="name of template database to use for PostgreSQL "
        "CREATE DATABASE (defaults to current database)",
    )
    make_option(
        "--oracledb-thick-mode",
        action="store_true",
        help="enables the 'thick mode' when testing with oracle+oracledb",
    )


def configure_follower(follower_ident):
    """Configure required state for a follower.

    This invokes in the parent process and typically includes
    database creation.

    """
    from sqlalchemy.testing import provision

    provision.FOLLOWER_IDENT = follower_ident


def memoize_important_follower_config(dict_):
    """Store important configuration we will need to send to a follower.

    This invokes in the parent process after normal config is set up.

    Hook is currently not used.

    """


def restore_important_follower_config(dict_):
    """Restore important configuration needed by a follower.

    This invokes in the follower process.

    Hook is currently not used.

    """


def read_config(root_path):
    global file_config
    file_config = configparser.ConfigParser()
    file_config.read(
        [str(root_path / "setup.cfg"), str(root_path / "test.cfg")]
    )


def pre_begin(opt):
    """things to set up early, before coverage might be setup."""
    global options
    options = opt
    for fn in pre_configure:
        fn(options, file_config)


def set_coverage_flag(value):
    options.has_coverage = value


def post_begin():
    """things to set up later, once we know coverage is running."""
    # Lazy setup of other options (post coverage)
    for fn in post_configure:
        fn(options, file_config)

    # late imports, has to happen after config.
    global util, fixtures, engines, exclusions, assertions, provision
    global warnings, profiling, config, testing
    from sqlalchemy import testing  # noqa
    from sqlalchemy.testing import fixtures, engines, exclusions  # noqa
    from sqlalchemy.testing import assertions, warnings, profiling  # noqa
    from sqlalchemy.testing import config, provision  # noqa
    from sqlalchemy import util  # noqa

    warnings.setup_filters()


def _log(opt_str, value, parser):
    global logging
    if not logging:
        import logging

        logging.basicConfig()

    if opt_str.endswith("-info"):
        logging.getLogger(value).setLevel(logging.INFO)
    elif opt_str.endswith("-debug"):
        logging.getLogger(value).setLevel(logging.DEBUG)


def _list_dbs(*args):
    if file_config is None:
        # assume the current working directory is the one containing the
        # setup file
        read_config(Path.cwd())
    print("Available --db options (use --dburi to override)")
    for macro in sorted(file_config.options("db")):
        print("%20s\t%s" % (macro, file_config.get("db", macro)))
    sys.exit(0)


def _requirements_opt(opt_str, value, parser):
    _setup_requirements(value)


def _set_tag_include(tag):
    def _do_include_tag(opt_str, value, parser):
        _include_tag(opt_str, tag, parser)

    return _do_include_tag


def _set_tag_exclude(tag):
    def _do_exclude_tag(opt_str, value, parser):
        _exclude_tag(opt_str, tag, parser)

    return _do_exclude_tag


def _exclude_tag(opt_str, value, parser):
    exclude_tags.add(value.replace("-", "_"))


def _include_tag(opt_str, value, parser):
    include_tags.add(value.replace("-", "_"))


pre_configure = []
post_configure = []


def pre(fn):
    pre_configure.append(fn)
    return fn


def post(fn):
    post_configure.append(fn)
    return fn


@pre
def _setup_options(opt, file_config):
    global options
    options = opt


@pre
def _register_sqlite_numeric_dialect(opt, file_config):
    from sqlalchemy.dialects import registry

    registry.register(
        "sqlite.pysqlite_numeric",
        "sqlalchemy.dialects.sqlite.pysqlite",
        "_SQLiteDialect_pysqlite_numeric",
    )
    registry.register(
        "sqlite.pysqlite_dollar",
        "sqlalchemy.dialects.sqlite.pysqlite",
        "_SQLiteDialect_pysqlite_dollar",
    )


@post
def __ensure_cext(opt, file_config):
    if os.environ.get("REQUIRE_SQLALCHEMY_CEXT", "0") == "1":
        from sqlalchemy.util import has_compiled_ext

        try:
            has_compiled_ext(raise_=True)
        except ImportError as err:
            raise AssertionError(
                "REQUIRE_SQLALCHEMY_CEXT is set but can't import the "
                "cython extensions"
            ) from err


@post
def _init_symbols(options, file_config):
    from sqlalchemy.testing import config

    config._fixture_functions = _fixture_fn_class()


@pre
def _set_disable_asyncio(opt, file_config):
    if opt.disable_asyncio:
        asyncio.ENABLE_ASYNCIO = False


@post
def _engine_uri(options, file_config):
    from sqlalchemy import testing
    from sqlalchemy.testing import config
    from sqlalchemy.testing import provision
    from sqlalchemy.engine import url as sa_url

    if options.dburi:
        db_urls = list(options.dburi)
    else:
        db_urls = []

    extra_drivers = options.dbdriver or []

    if options.db:
        for db_token in options.db:
            for db in re.split(r"[,\s]+", db_token):
                if db not in file_config.options("db"):
                    raise RuntimeError(
                        "Unknown URI specifier '%s'.  "
                        "Specify --dbs for known uris." % db
                    )
                else:
                    db_urls.append(file_config.get("db", db))

    if not db_urls:
        db_urls.append(file_config.get("db", "default"))

    config._current = None

    if options.write_idents and provision.FOLLOWER_IDENT:
        for db_url in [sa_url.make_url(db_url) for db_url in db_urls]:
            with open(options.write_idents, "a") as file_:
                file_.write(
                    f"{provision.FOLLOWER_IDENT} "
                    f"{db_url.render_as_string(hide_password=False)}\n"
                )

    expanded_urls = list(provision.generate_db_urls(db_urls, extra_drivers))

    for db_url in expanded_urls:
        log.info("Adding database URL: %s", db_url)

        cfg = provision.setup_config(
            db_url, options, file_config, provision.FOLLOWER_IDENT
        )
        if not config._current:
            cfg.set_as_current(cfg, testing)


@post
def _requirements(options, file_config):
    requirement_cls = file_config.get("sqla_testing", "requirement_cls")
    _setup_requirements(requirement_cls)


def _setup_requirements(argument):
    from sqlalchemy.testing import config
    from sqlalchemy import testing

    modname, clsname = argument.split(":")

    # importlib.import_module() only introduced in 2.7, a little
    # late
    mod = __import__(modname)
    for component in modname.split(".")[1:]:
        mod = getattr(mod, component)
    req_cls = getattr(mod, clsname)

    config.requirements = testing.requires = req_cls()

    config.bootstrapped_as_sqlalchemy = bootstrapped_as_sqlalchemy


@post
def _prep_testing_database(options, file_config):
    from sqlalchemy.testing import config

    if options.dropfirst:
        from sqlalchemy.testing import provision

        for cfg in config.Config.all_configs():
            provision.drop_all_schema_objects(cfg, cfg.db)


@post
def _post_setup_options(opt, file_config):
    from sqlalchemy.testing import config

    config.options = options
    config.file_config = file_config


@post
def _setup_profiling(options, file_config):
    from sqlalchemy.testing import profiling

    profiling._profile_stats = profiling.ProfileStatsFile(
        file_config.get("sqla_testing", "profile_file"),
        sort=options.profilesort,
        dump=options.profiledump,
    )


def want_class(name, cls):
    if not issubclass(cls, fixtures.TestBase):
        return False
    elif name.startswith("_"):
        return False
    else:
        return True


def want_method(cls, fn):
    if not fn.__name__.startswith("test_"):
        return False
    elif fn.__module__ is None:
        return False
    else:
        return True


def generate_sub_tests(cls, module, markers):
    if "backend" in markers or "sparse_backend" in markers:
        sparse = "sparse_backend" in markers
        for cfg in _possible_configs_for_cls(cls, sparse=sparse):
            orig_name = cls.__name__

            # we can have special chars in these names except for the
            # pytest junit plugin, which is tripped up by the brackets
            # and periods, so sanitize

            alpha_name = re.sub(r"[_\[\]\.]+", "_", cfg.name)
            alpha_name = re.sub(r"_+$", "", alpha_name)
            name = "%s_%s" % (cls.__name__, alpha_name)
            subcls = type(
                name,
                (cls,),
                {"_sa_orig_cls_name": orig_name, "__only_on_config__": cfg},
            )
            setattr(module, name, subcls)
            yield subcls
    else:
        yield cls


def start_test_class_outside_fixtures(cls):
    _do_skips(cls)
    _setup_engine(cls)


def stop_test_class(cls):
    # close sessions, immediate connections, etc.
    fixtures.stop_test_class_inside_fixtures(cls)

    # close outstanding connection pool connections, dispose of
    # additional engines
    engines.testing_reaper.stop_test_class_inside_fixtures()


def stop_test_class_outside_fixtures(cls):
    engines.testing_reaper.stop_test_class_outside_fixtures()
    provision.stop_test_class_outside_fixtures(config, config.db, cls)
    try:
        if not options.low_connections:
            assertions.global_cleanup_assertions()
    finally:
        _restore_engine()


def _restore_engine():
    if config._current:
        config._current.reset(testing)


def final_process_cleanup():
    engines.testing_reaper.final_cleanup()
    assertions.global_cleanup_assertions()
    _restore_engine()


def _setup_engine(cls):
    if getattr(cls, "__engine_options__", None):
        opts = dict(cls.__engine_options__)
        opts["scope"] = "class"
        eng = engines.testing_engine(options=opts)
        config._current.push_engine(eng, testing)


def before_test(test, test_module_name, test_class, test_name):
    # format looks like:
    # "test.aaa_profiling.test_compiler.CompileTest.test_update_whereclause"

    name = getattr(test_class, "_sa_orig_cls_name", test_class.__name__)

    id_ = "%s.%s.%s" % (test_module_name, name, test_name)

    profiling._start_current_test(id_)


def after_test(test):
    fixtures.after_test()
    engines.testing_reaper.after_test()


def after_test_fixtures(test):
    engines.testing_reaper.after_test_outside_fixtures(test)


def _possible_configs_for_cls(cls, reasons=None, sparse=False):
    all_configs = set(config.Config.all_configs())

    if cls.__unsupported_on__:
        spec = exclusions.db_spec(*cls.__unsupported_on__)
        for config_obj in list(all_configs):
            if spec(config_obj):
                all_configs.remove(config_obj)

    if getattr(cls, "__only_on__", None):
        spec = exclusions.db_spec(*util.to_list(cls.__only_on__))
        for config_obj in list(all_configs):
            if not spec(config_obj):
                all_configs.remove(config_obj)

    if getattr(cls, "__only_on_config__", None):
        all_configs.intersection_update([cls.__only_on_config__])

    if hasattr(cls, "__requires__"):
        requirements = config.requirements
        for config_obj in list(all_configs):
            for requirement in cls.__requires__:
                check = getattr(requirements, requirement)

                skip_reasons = check.matching_config_reasons(config_obj)
                if skip_reasons:
                    all_configs.remove(config_obj)
                    if reasons is not None:
                        reasons.extend(skip_reasons)
                    break

    if hasattr(cls, "__prefer_requires__"):
        non_preferred = set()
        requirements = config.requirements
        for config_obj in list(all_configs):
            for requirement in cls.__prefer_requires__:
                check = getattr(requirements, requirement)

                if not check.enabled_for_config(config_obj):
                    non_preferred.add(config_obj)
        if all_configs.difference(non_preferred):
            all_configs.difference_update(non_preferred)

    if sparse:
        # pick only one config from each base dialect
        # sorted so we get the same backend each time selecting the highest
        # server version info.
        per_dialect = {}
        for cfg in reversed(
            sorted(
                all_configs,
                key=lambda cfg: (
                    cfg.db.name,
                    cfg.db.driver,
                    cfg.db.dialect.server_version_info,
                ),
            )
        ):
            db = cfg.db.name
            if db not in per_dialect:
                per_dialect[db] = cfg
        return per_dialect.values()

    return all_configs


def _do_skips(cls):
    reasons = []
    all_configs = _possible_configs_for_cls(cls, reasons)

    if getattr(cls, "__skip_if__", False):
        for c in getattr(cls, "__skip_if__"):
            if c():
                config.skip_test(
                    "'%s' skipped by %s" % (cls.__name__, c.__name__)
                )

    if not all_configs:
        msg = "'%s.%s' unsupported on any DB implementation %s%s" % (
            cls.__module__,
            cls.__name__,
            ", ".join(
                "'%s(%s)+%s'"
                % (
                    config_obj.db.name,
                    ".".join(
                        str(dig)
                        for dig in exclusions._server_version(config_obj.db)
                    ),
                    config_obj.db.driver,
                )
                for config_obj in config.Config.all_configs()
            ),
            ", ".join(reasons),
        )
        config.skip_test(msg)
    elif hasattr(cls, "__prefer_backends__"):
        non_preferred = set()
        spec = exclusions.db_spec(*util.to_list(cls.__prefer_backends__))
        for config_obj in all_configs:
            if not spec(config_obj):
                non_preferred.add(config_obj)
        if all_configs.difference(non_preferred):
            all_configs.difference_update(non_preferred)

    if config._current not in all_configs:
        _setup_config(all_configs.pop(), cls)


def _setup_config(config_obj, ctx):
    config._current.push(config_obj, testing)


class FixtureFunctions(abc.ABC):
    @abc.abstractmethod
    def skip_test_exception(self, *arg, **kw):
        raise NotImplementedError()

    @abc.abstractmethod
    def combinations(self, *args, **kw):
        raise NotImplementedError()

    @abc.abstractmethod
    def param_ident(self, *args, **kw):
        raise NotImplementedError()

    @abc.abstractmethod
    def fixture(self, *arg, **kw):
        raise NotImplementedError()

    def get_current_test_name(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def mark_base_test_class(self) -> Any:
        raise NotImplementedError()

    @abc.abstractproperty
    def add_to_marker(self):
        raise NotImplementedError()


_fixture_fn_class = None


def set_fixture_functions(fixture_fn_class):
    global _fixture_fn_class
    _fixture_fn_class = fixture_fn_class

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\fortran.py ===
"""
Fortran code printer

The FCodePrinter converts single SymPy expressions into single Fortran
expressions, using the functions defined in the Fortran 77 standard where
possible. Some useful pointers to Fortran can be found on wikipedia:

https://en.wikipedia.org/wiki/Fortran

Most of the code below is based on the "Professional Programmer\'s Guide to
Fortran77" by Clive G. Page:

https://www.star.le.ac.uk/~cgp/prof77.html

Fortran is a case-insensitive language. This might cause trouble because
SymPy is case sensitive. So, fcode adds underscores to variable names when
it is necessary to make them different for Fortran.
"""

from __future__ import annotations
from typing import Any

from collections import defaultdict
from itertools import chain
import string

from sympy.codegen.ast import (
    Assignment, Declaration, Pointer, value_const,
    float32, float64, float80, complex64, complex128, int8, int16, int32,
    int64, intc, real, integer,  bool_, complex_, none, stderr, stdout
)
from sympy.codegen.fnodes import (
    allocatable, isign, dsign, cmplx, merge, literal_dp, elemental, pure,
    intent_in, intent_out, intent_inout
)
from sympy.core import S, Add, N, Float, Symbol
from sympy.core.function import Function
from sympy.core.numbers import equal_valued
from sympy.core.relational import Eq
from sympy.sets import Range
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.printing.printer import printer_context

# These are defined in the other file so we can avoid importing sympy.codegen
# from the top-level 'import sympy'. Export them here as well.
from sympy.printing.codeprinter import fcode, print_fcode # noqa:F401

known_functions = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "log": "log",
    "exp": "exp",
    "erf": "erf",
    "Abs": "abs",
    "conjugate": "conjg",
    "Max": "max",
    "Min": "min",
}


class FCodePrinter(CodePrinter):
    """A printer to convert SymPy expressions to strings of Fortran code"""
    printmethod = "_fcode"
    language = "Fortran"

    type_aliases = {
        integer: int32,
        real: float64,
        complex_: complex128,
    }

    type_mappings = {
        intc: 'integer(c_int)',
        float32: 'real*4',  # real(kind(0.e0))
        float64: 'real*8',  # real(kind(0.d0))
        float80: 'real*10', # real(kind(????))
        complex64: 'complex*8',
        complex128: 'complex*16',
        int8: 'integer*1',
        int16: 'integer*2',
        int32: 'integer*4',
        int64: 'integer*8',
        bool_: 'logical'
    }

    type_modules = {
        intc: {'iso_c_binding': 'c_int'}
    }

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 17,
        'user_functions': {},
        'source_format': 'fixed',
        'contract': True,
        'standard': 77,
        'name_mangling': True,
    })

    _operators = {
        'and': '.and.',
        'or': '.or.',
        'xor': '.neqv.',
        'equivalent': '.eqv.',
        'not': '.not. ',
    }

    _relationals = {
        '!=': '/=',
    }

    def __init__(self, settings=None):
        if not settings:
            settings = {}
        self.mangled_symbols = {}         # Dict showing mapping of all words
        self.used_name = []
        self.type_aliases = dict(chain(self.type_aliases.items(),
                                       settings.pop('type_aliases', {}).items()))
        self.type_mappings = dict(chain(self.type_mappings.items(),
                                        settings.pop('type_mappings', {}).items()))
        super().__init__(settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)
        # leading columns depend on fixed or free format
        standards = {66, 77, 90, 95, 2003, 2008}
        if self._settings['standard'] not in standards:
            raise ValueError("Unknown Fortran standard: %s" % self._settings[
                             'standard'])
        self.module_uses = defaultdict(set)  # e.g.: use iso_c_binding, only: c_int

    @property
    def _lead(self):
        if self._settings['source_format'] == 'fixed':
            return {'code': "      ", 'cont': "     @ ", 'comment': "C     "}
        elif self._settings['source_format'] == 'free':
            return {'code': "", 'cont': "      ", 'comment': "! "}
        else:
            raise ValueError("Unknown source format: %s" % self._settings['source_format'])

    def _print_Symbol(self, expr):
        if self._settings['name_mangling'] == True:
            if expr not in self.mangled_symbols:
                name = expr.name
                while name.lower() in self.used_name:
                    name += '_'
                self.used_name.append(name.lower())
                if name == expr.name:
                    self.mangled_symbols[expr] = expr
                else:
                    self.mangled_symbols[expr] = Symbol(name)

            expr = expr.xreplace(self.mangled_symbols)

        name = super()._print_Symbol(expr)
        return name

    def _rate_index_position(self, p):
        return -p*5

    def _get_statement(self, codestring):
        return codestring

    def _get_comment(self, text):
        return "! {}".format(text)

    def _declare_number_const(self, name, value):
        return "parameter ({} = {})".format(name, self._print(value))

    def _print_NumberSymbol(self, expr):
        # A Number symbol that is not implemented here or with _printmethod
        # is registered and evaluated
        self._number_symbols.add((expr, Float(expr.evalf(self._settings['precision']))))
        return str(expr)

    def _format_code(self, lines):
        return self._wrap_fortran(self.indent_code(lines))

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for j in range(cols) for i in range(rows))

    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        for i in indices:
            # fortran arrays start at 1 and end at dimension
            var, start, stop = map(self._print,
                    [i.label, i.lower + 1, i.upper + 1])
            open_lines.append("do %s = %s, %s" % (var, start, stop))
            close_lines.append("end do")
        return open_lines, close_lines

    def _print_sign(self, expr):
        from sympy.functions.elementary.complexes import Abs
        arg, = expr.args
        if arg.is_integer:
            new_expr = merge(0, isign(1, arg), Eq(arg, 0))
        elif (arg.is_complex or arg.is_infinite):
            new_expr = merge(cmplx(literal_dp(0), literal_dp(0)), arg/Abs(arg), Eq(Abs(arg), literal_dp(0)))
        else:
            new_expr = merge(literal_dp(0), dsign(literal_dp(1), arg), Eq(arg, literal_dp(0)))
        return self._print(new_expr)


    def _print_Piecewise(self, expr):
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
                    lines.append("if (%s) then" % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else")
                else:
                    lines.append("else if (%s) then" % self._print(c))
                lines.append(self._print(e))
            lines.append("end if")
            return "\n".join(lines)
        elif self._settings["standard"] >= 95:
            # Only supported in F95 and newer:
            # The piecewise was used in an expression, need to do inline
            # operators. This has the downside that inline operators will
            # not work for statements that span multiple lines (Matrix or
            # Indexed expressions).
            pattern = "merge({T}, {F}, {COND})"
            code = self._print(expr.args[-1].expr)
            terms = list(expr.args[:-1])
            while terms:
                e, c = terms.pop()
                expr = self._print(e)
                cond = self._print(c)
                code = pattern.format(T=expr, F=code, COND=cond)
            return code
        else:
            # `merge` is not supported prior to F95
            raise NotImplementedError("Using Piecewise as an expression using "
                                      "inline operators is not supported in "
                                      "standards earlier than Fortran95.")

    def _print_MatrixElement(self, expr):
        return "{}({}, {})".format(self.parenthesize(expr.parent,
                PRECEDENCE["Atom"], strict=True), expr.i + 1, expr.j + 1)

    def _print_Add(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        # collect the purely real and purely imaginary parts:
        pure_real = []
        pure_imaginary = []
        mixed = []
        for arg in expr.args:
            if arg.is_number and arg.is_real:
                pure_real.append(arg)
            elif arg.is_number and arg.is_imaginary:
                pure_imaginary.append(arg)
            else:
                mixed.append(arg)
        if pure_imaginary:
            if mixed:
                PREC = precedence(expr)
                term = Add(*mixed)
                t = self._print(term)
                if t.startswith('-'):
                    sign = "-"
                    t = t[1:]
                else:
                    sign = "+"
                if precedence(term) < PREC:
                    t = "(%s)" % t

                return "cmplx(%s,%s) %s %s" % (
                    self._print(Add(*pure_real)),
                    self._print(-S.ImaginaryUnit*Add(*pure_imaginary)),
                    sign, t,
                )
            else:
                return "cmplx(%s,%s)" % (
                    self._print(Add(*pure_real)),
                    self._print(-S.ImaginaryUnit*Add(*pure_imaginary)),
                )
        else:
            return CodePrinter._print_Add(self, expr)

    def _print_Function(self, expr):
        # All constant function args are evaluated as floats
        prec =  self._settings['precision']
        args = [N(a, prec) for a in expr.args]
        eval_expr = expr.func(*args)
        if not isinstance(eval_expr, Function):
            return self._print(eval_expr)
        else:
            return CodePrinter._print_Function(self, expr.func(*args))

    def _print_Mod(self, expr):
        # NOTE : Fortran has the functions mod() and modulo(). modulo() behaves
        # the same wrt to the sign of the arguments as Python and SymPy's
        # modulus computations (% and Mod()) but is not available in Fortran 66
        # or Fortran 77, thus we raise an error.
        if self._settings['standard'] in [66, 77]:
            msg = ("Python % operator and SymPy's Mod() function are not "
                   "supported by Fortran 66 or 77 standards.")
            raise NotImplementedError(msg)
        else:
            x, y = expr.args
            return "      modulo({}, {})".format(self._print(x), self._print(y))

    def _print_ImaginaryUnit(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        return "cmplx(0,1)"

    def _print_int(self, expr):
        return str(expr)

    def _print_Mul(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        if expr.is_number and expr.is_imaginary:
            return "cmplx(0,%s)" % (
                self._print(-S.ImaginaryUnit*expr)
            )
        else:
            return CodePrinter._print_Mul(self, expr)

    def _print_Pow(self, expr):
        PREC = precedence(expr)
        if equal_valued(expr.exp, -1):
            return '%s/%s' % (
                self._print(literal_dp(1)),
                self.parenthesize(expr.base, PREC)
            )
        elif equal_valued(expr.exp, 0.5):
            if expr.base.is_integer:
                # Fortran intrinsic sqrt() does not accept integer argument
                if expr.base.is_Number:
                    return 'sqrt(%s.0d0)' % self._print(expr.base)
                else:
                    return 'sqrt(dble(%s))' % self._print(expr.base)
            else:
                return 'sqrt(%s)' % self._print(expr.base)
        else:
            return CodePrinter._print_Pow(self, expr)

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return "%d.0d0/%d.0d0" % (p, q)

    def _print_Float(self, expr):
        printed = CodePrinter._print_Float(self, expr)
        e = printed.find('e')
        if e > -1:
            return "%sd%s" % (printed[:e], printed[e + 1:])
        return "%sd0" % printed

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        op = op if op not in self._relationals else self._relationals[op]
        return "{} {} {}".format(lhs_code, op, rhs_code)

    def _print_Indexed(self, expr):
        inds = [ self._print(i) for i in expr.indices ]
        return "%s(%s)" % (self._print(expr.base.label), ", ".join(inds))

    def _print_AugmentedAssignment(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        return self._get_statement("{0} = {0} {1} {2}".format(
            self._print(lhs_code), self._print(expr.binop), self._print(rhs_code)))

    def _print_sum_(self, sm):
        params = self._print(sm.array)
        if sm.dim != None: # Must use '!= None', cannot use 'is not None'
            params += ', ' + self._print(sm.dim)
        if sm.mask != None: # Must use '!= None', cannot use 'is not None'
            params += ', mask=' + self._print(sm.mask)
        return '%s(%s)' % (sm.__class__.__name__.rstrip('_'), params)

    def _print_product_(self, prod):
        return self._print_sum_(prod)

    def _print_Do(self, do):
        excl = ['concurrent']
        if do.step == 1:
            excl.append('step')
            step = ''
        else:
            step = ', {step}'

        return (
            'do {concurrent}{counter} = {first}, {last}'+step+'\n'
            '{body}\n'
            'end do\n'
        ).format(
            concurrent='concurrent ' if do.concurrent else '',
            **do.kwargs(apply=lambda arg: self._print(arg), exclude=excl)
        )

    def _print_ImpliedDoLoop(self, idl):
        step = '' if idl.step == 1 else ', {step}'
        return ('({expr}, {counter} = {first}, {last}'+step+')').format(
            **idl.kwargs(apply=lambda arg: self._print(arg))
        )

    def _print_For(self, expr):
        target = self._print(expr.target)
        if isinstance(expr.iterable, Range):
            start, stop, step = expr.iterable.args
        else:
            raise NotImplementedError("Only iterable currently supported is Range")
        body = self._print(expr.body)
        return ('do {target} = {start}, {stop}, {step}\n'
                '{body}\n'
                'end do').format(target=target, start=start, stop=stop - 1,
                        step=step, body=body)

    def _print_Type(self, type_):
        type_ = self.type_aliases.get(type_, type_)
        type_str = self.type_mappings.get(type_, type_.name)
        module_uses = self.type_modules.get(type_)
        if module_uses:
            for k, v in module_uses:
                self.module_uses[k].add(v)
        return type_str

    def _print_Element(self, elem):
        return '{symbol}({idxs})'.format(
            symbol=self._print(elem.symbol),
            idxs=', '.join((self._print(arg) for arg in elem.indices))
        )

    def _print_Extent(self, ext):
        return str(ext)

    def _print_Declaration(self, expr):
        var = expr.variable
        val = var.value
        dim = var.attr_params('dimension')
        intents = [intent in var.attrs for intent in (intent_in, intent_out, intent_inout)]
        if intents.count(True) == 0:
            intent = ''
        elif intents.count(True) == 1:
            intent = ', intent(%s)' % ['in', 'out', 'inout'][intents.index(True)]
        else:
            raise ValueError("Multiple intents specified for %s" % self)

        if isinstance(var, Pointer):
            raise NotImplementedError("Pointers are not available by default in Fortran.")
        if self._settings["standard"] >= 90:
            result = '{t}{vc}{dim}{intent}{alloc} :: {s}'.format(
                t=self._print(var.type),
                vc=', parameter' if value_const in var.attrs else '',
                dim=', dimension(%s)' % ', '.join((self._print(arg) for arg in dim)) if dim else '',
                intent=intent,
                alloc=', allocatable' if allocatable in var.attrs else '',
                s=self._print(var.symbol)
            )
            if val != None: # Must be "!= None", cannot be "is not None"
                result += ' = %s' % self._print(val)
        else:
            if value_const in var.attrs or val:
                raise NotImplementedError("F77 init./parameter statem. req. multiple lines.")
            result = ' '.join((self._print(arg) for arg in [var.type, var.symbol]))

        return result


    def _print_Infinity(self, expr):
        return '(huge(%s) + 1)' % self._print(literal_dp(0))

    def _print_While(self, expr):
        return 'do while ({condition})\n{body}\nend do'.format(**expr.kwargs(
            apply=lambda arg: self._print(arg)))

    def _print_BooleanTrue(self, expr):
        return '.true.'

    def _print_BooleanFalse(self, expr):
        return '.false.'

    def _pad_leading_columns(self, lines):
        result = []
        for line in lines:
            if line.startswith('!'):
                result.append(self._lead['comment'] + line[1:].lstrip())
            else:
                result.append(self._lead['code'] + line)
        return result

    def _wrap_fortran(self, lines):
        """Wrap long Fortran lines

           Argument:
             lines  --  a list of lines (without \\n character)

           A comment line is split at white space. Code lines are split with a more
           complex rule to give nice results.
        """
        # routine to find split point in a code line
        my_alnum = set("_+-." + string.digits + string.ascii_letters)
        my_white = set(" \t()")

        def split_pos_code(line, endpos):
            if len(line) <= endpos:
                return len(line)
            pos = endpos
            split = lambda pos: \
                (line[pos] in my_alnum and line[pos - 1] not in my_alnum) or \
                (line[pos] not in my_alnum and line[pos - 1] in my_alnum) or \
                (line[pos] in my_white and line[pos - 1] not in my_white) or \
                (line[pos] not in my_white and line[pos - 1] in my_white)
            while not split(pos):
                pos -= 1
                if pos == 0:
                    return endpos
            return pos
        # split line by line and add the split lines to result
        result = []
        if self._settings['source_format'] == 'free':
            trailing = ' &'
        else:
            trailing = ''
        for line in lines:
            if line.startswith(self._lead['comment']):
                # comment line
                if len(line) > 72:
                    pos = line.rfind(" ", 6, 72)
                    if pos == -1:
                        pos = 72
                    hunk = line[:pos]
                    line = line[pos:].lstrip()
                    result.append(hunk)
                    while line:
                        pos = line.rfind(" ", 0, 66)
                        if pos == -1 or len(line) < 66:
                            pos = 66
                        hunk = line[:pos]
                        line = line[pos:].lstrip()
                        result.append("%s%s" % (self._lead['comment'], hunk))
                else:
                    result.append(line)
            elif line.startswith(self._lead['code']):
                # code line
                pos = split_pos_code(line, 72)
                hunk = line[:pos].rstrip()
                line = line[pos:].lstrip()
                if line:
                    hunk += trailing
                result.append(hunk)
                while line:
                    pos = split_pos_code(line, 65)
                    hunk = line[:pos].rstrip()
                    line = line[pos:].lstrip()
                    if line:
                        hunk += trailing
                    result.append("%s%s" % (self._lead['cont'], hunk))
            else:
                result.append(line)
        return result

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        free = self._settings['source_format'] == 'free'
        code = [ line.lstrip(' \t') for line in code ]

        inc_keyword = ('do ', 'if(', 'if ', 'do\n', 'else', 'program', 'interface')
        dec_keyword = ('end do', 'enddo', 'end if', 'endif', 'else', 'end program', 'end interface')

        increase = [ int(any(map(line.startswith, inc_keyword)))
                     for line in code ]
        decrease = [ int(any(map(line.startswith, dec_keyword)))
                     for line in code ]
        continuation = [ int(any(map(line.endswith, ['&', '&\n'])))
                         for line in code ]

        level = 0
        cont_padding = 0
        tabwidth = 3
        new_code = []
        for i, line in enumerate(code):
            if line in ('', '\n'):
                new_code.append(line)
                continue
            level -= decrease[i]

            if free:
                padding = " "*(level*tabwidth + cont_padding)
            else:
                padding = " "*level*tabwidth

            line = "%s%s" % (padding, line)
            if not free:
                line = self._pad_leading_columns([line])[0]

            new_code.append(line)

            if continuation[i]:
                cont_padding = 2*tabwidth
            else:
                cont_padding = 0
            level += increase[i]

        if not free:
            return self._wrap_fortran(new_code)
        return new_code

    def _print_GoTo(self, goto):
        if goto.expr:  # computed goto
            return "go to ({labels}), {expr}".format(
                labels=', '.join((self._print(arg) for arg in goto.labels)),
                expr=self._print(goto.expr)
            )
        else:
            lbl, = goto.labels
            return "go to %s" % self._print(lbl)

    def _print_Program(self, prog):
        return (
            "program {name}\n"
            "{body}\n"
            "end program\n"
        ).format(**prog.kwargs(apply=lambda arg: self._print(arg)))

    def _print_Module(self, mod):
        return (
            "module {name}\n"
            "{declarations}\n"
            "\ncontains\n\n"
            "{definitions}\n"
            "end module\n"
        ).format(**mod.kwargs(apply=lambda arg: self._print(arg)))

    def _print_Stream(self, strm):
        if strm.name == 'stdout' and self._settings["standard"] >= 2003:
            self.module_uses['iso_c_binding'].add('stdint=>input_unit')
            return 'input_unit'
        elif strm.name == 'stderr' and self._settings["standard"] >= 2003:
            self.module_uses['iso_c_binding'].add('stdint=>error_unit')
            return 'error_unit'
        else:
            if strm.name == 'stdout':
                return '*'
            else:
                return strm.name

    def _print_Print(self, ps):
        if ps.format_string == none: # Must be '!= None', cannot be 'is not None'
            template = "print {fmt}, {iolist}"
            fmt = '*'
        else:
            template = 'write(%(out)s, fmt="{fmt}", advance="no"), {iolist}' % {
                'out': {stderr: '0', stdout: '6'}.get(ps.file, '*')
            }
            fmt = self._print(ps.format_string)
        return template.format(fmt=fmt, iolist=', '.join(
            (self._print(arg) for arg in ps.print_args)))

    def _print_Return(self, rs):
        arg, = rs.args
        return "{result_name} = {arg}".format(
            result_name=self._context.get('result_name', 'sympy_result'),
            arg=self._print(arg)
        )

    def _print_FortranReturn(self, frs):
        arg, = frs.args
        if arg:
            return 'return %s' % self._print(arg)
        else:
            return 'return'

    def _head(self, entity, fp, **kwargs):
        bind_C_params = fp.attr_params('bind_C')
        if bind_C_params is None:
            bind = ''
        else:
            bind = ' bind(C, name="%s")' % bind_C_params[0] if bind_C_params else ' bind(C)'
        result_name = self._settings.get('result_name', None)
        return (
            "{entity}{name}({arg_names}){result}{bind}\n"
            "{arg_declarations}"
        ).format(
            entity=entity,
            name=self._print(fp.name),
            arg_names=', '.join([self._print(arg.symbol) for arg in fp.parameters]),
            result=(' result(%s)' % result_name) if result_name else '',
            bind=bind,
            arg_declarations='\n'.join((self._print(Declaration(arg)) for arg in fp.parameters))
        )

    def _print_FunctionPrototype(self, fp):
        entity = "{} function ".format(self._print(fp.return_type))
        return (
            "interface\n"
            "{function_head}\n"
            "end function\n"
            "end interface"
        ).format(function_head=self._head(entity, fp))

    def _print_FunctionDefinition(self, fd):
        if elemental in fd.attrs:
            prefix = 'elemental '
        elif pure in fd.attrs:
            prefix = 'pure '
        else:
            prefix = ''

        entity = "{} function ".format(self._print(fd.return_type))
        with printer_context(self, result_name=fd.name):
            return (
                "{prefix}{function_head}\n"
                "{body}\n"
                "end function\n"
            ).format(
                prefix=prefix,
                function_head=self._head(entity, fd),
                body=self._print(fd.body)
            )

    def _print_Subroutine(self, sub):
        return (
            '{subroutine_head}\n'
            '{body}\n'
            'end subroutine\n'
        ).format(
            subroutine_head=self._head('subroutine ', sub),
            body=self._print(sub.body)
        )

    def _print_SubroutineCall(self, scall):
        return 'call {name}({args})'.format(
            name=self._print(scall.name),
            args=', '.join((self._print(arg) for arg in scall.subroutine_args))
        )

    def _print_use_rename(self, rnm):
        return "%s => %s" % tuple((self._print(arg) for arg in rnm.args))

    def _print_use(self, use):
        result = 'use %s' % self._print(use.namespace)
        if use.rename != None: # Must be '!= None', cannot be 'is not None'
            result += ', ' + ', '.join([self._print(rnm) for rnm in use.rename])
        if use.only != None: # Must be '!= None', cannot be 'is not None'
            result += ', only: ' + ', '.join([self._print(nly) for nly in use.only])
        return result

    def _print_BreakToken(self, _):
        return 'exit'

    def _print_ContinueToken(self, _):
        return 'cycle'

    def _print_ArrayConstructor(self, ac):
        fmtstr = "[%s]" if self._settings["standard"] >= 2003 else '(/%s/)'
        return fmtstr % ', '.join((self._print(arg) for arg in ac.elements))

    def _print_ArrayElement(self, elem):
        return '{symbol}({idxs})'.format(
            symbol=self._print(elem.name),
            idxs=', '.join((self._print(arg) for arg in elem.indices))
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\stable_diffusion\demo_utils.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Modified from TensorRT demo diffusion, which has the following license:
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
import argparse
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import controlnet_aux
import cv2
import numpy as np
import torch
from cuda import cudart
from diffusion_models import PipelineInfo
from engine_builder import EngineType, get_engine_paths, get_engine_type
from PIL import Image
from pipeline_stable_diffusion import StableDiffusionPipeline


class RawTextArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def arg_parser(description: str):
    return argparse.ArgumentParser(
        description=description,
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
    )


def set_default_arguments(args):
    # set default value for some arguments if not provided
    if args.height is None:
        args.height = PipelineInfo.default_resolution(args.version)

    if args.width is None:
        args.width = PipelineInfo.default_resolution(args.version)

    is_lcm = (args.version == "xl-1.0" and args.lcm) or "lcm" in args.lora_weights
    is_turbo = args.version in ["sd-turbo", "xl-turbo"]
    if args.denoising_steps is None:
        args.denoising_steps = 4 if is_turbo else 8 if is_lcm else (30 if args.version == "xl-1.0" else 50)

    if args.scheduler is None:
        args.scheduler = "LCM" if (is_lcm or is_turbo) else ("EulerA" if args.version == "xl-1.0" else "DDIM")

    if args.guidance is None:
        args.guidance = 0.0 if (is_lcm or is_turbo) else (5.0 if args.version == "xl-1.0" else 7.5)


def parse_arguments(is_xl: bool, parser):
    engines = ["ORT_CUDA", "ORT_TRT", "TRT", "TORCH"]

    parser.add_argument(
        "-e",
        "--engine",
        type=str,
        default=engines[0],
        choices=engines,
        help="Backend engine in {engines}. "
        "ORT_CUDA is CUDA execution provider; ORT_TRT is Tensorrt execution provider; TRT is TensorRT",
    )

    supported_versions = PipelineInfo.supported_versions(is_xl)
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="xl-1.0" if is_xl else "1.5",
        choices=supported_versions,
        help="Version of Stable Diffusion" + (" XL." if is_xl else "."),
    )

    parser.add_argument(
        "-y",
        "--height",
        type=int,
        default=None,
        help="Height of image to generate (must be multiple of 8).",
    )
    parser.add_argument(
        "-x", "--width", type=int, default=None, help="Height of image to generate (must be multiple of 8)."
    )

    parser.add_argument(
        "-s",
        "--scheduler",
        type=str,
        default=None,
        choices=["DDIM", "EulerA", "UniPC", "LCM"],
        help="Scheduler for diffusion process" + " of base" if is_xl else "",
    )

    parser.add_argument(
        "-wd",
        "--work-dir",
        default=".",
        help="Root Directory to store torch or ONNX models, built engines and output images etc.",
    )

    parser.add_argument(
        "-i",
        "--engine-dir",
        default=None,
        help="Root Directory to store built engines or optimized ONNX models etc.",
    )

    parser.add_argument("prompt", nargs="*", default=[""], help="Text prompt(s) to guide image generation.")

    parser.add_argument(
        "-n",
        "--negative-prompt",
        nargs="*",
        default=[""],
        help="Optional negative prompt(s) to guide the image generation.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        choices=[1, 2, 4, 8, 16],
        help="Number of times to repeat the prompt (batch size multiplier).",
    )

    parser.add_argument(
        "-d",
        "--denoising-steps",
        type=int,
        default=None,
        help="Number of denoising steps" + (" in base." if is_xl else "."),
    )

    parser.add_argument(
        "-g",
        "--guidance",
        type=float,
        default=None,
        help="Higher guidance scale encourages to generate images that are closely linked to the text prompt.",
    )

    parser.add_argument(
        "-ls", "--lora-scale", type=float, default=1, help="Scale of LoRA weights, default 1 (must between 0 and 1)"
    )
    parser.add_argument("-lw", "--lora-weights", type=str, default="", help="LoRA weights to apply in the base model")

    if is_xl:
        parser.add_argument(
            "--lcm",
            action="store_true",
            help="Use fine-tuned latent consistency model to replace the UNet in base.",
        )

        parser.add_argument(
            "-rs",
            "--refiner-scheduler",
            type=str,
            default="EulerA",
            choices=["DDIM", "EulerA", "UniPC"],
            help="Scheduler for diffusion process of refiner.",
        )

        parser.add_argument(
            "-rg",
            "--refiner-guidance",
            type=float,
            default=5.0,
            help="Guidance scale used in refiner.",
        )

        parser.add_argument(
            "-rd",
            "--refiner-denoising-steps",
            type=int,
            default=30,
            help="Number of denoising steps in refiner. Note that actual steps is refiner_denoising_steps * strength.",
        )

        parser.add_argument(
            "--strength",
            type=float,
            default=0.3,
            help="A value between 0 and 1. The higher the value less the final image similar to the seed image.",
        )

        parser.add_argument(
            "-r",
            "--enable-refiner",
            action="store_true",
            help="Enable SDXL refiner to refine image from base pipeline.",
        )

    # ONNX export
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=None,
        choices=range(14, 18),
        help="Select ONNX opset version to target for exported models.",
    )

    # Engine build options.
    parser.add_argument(
        "-db",
        "--build-dynamic-batch",
        action="store_true",
        help="Build TensorRT engines to support dynamic batch size.",
    )
    parser.add_argument(
        "-ds",
        "--build-dynamic-shape",
        action="store_true",
        help="Build TensorRT engines to support dynamic image sizes.",
    )
    parser.add_argument("--max-batch-size", type=int, default=None, choices=[1, 2, 4, 8, 16, 32], help="Max batch size")

    # Inference related options
    parser.add_argument(
        "-nw", "--num-warmup-runs", type=int, default=5, help="Number of warmup runs before benchmarking performance."
    )
    parser.add_argument("--nvtx-profile", action="store_true", help="Enable NVTX markers for performance profiling.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random generator to get consistent results.")
    parser.add_argument("--deterministic", action="store_true", help="use deterministic algorithms.")
    parser.add_argument("-dc", "--disable-cuda-graph", action="store_true", help="Disable cuda graph.")

    parser.add_argument("--framework-model-dir", default=None, help="framework model directory")

    group = parser.add_argument_group("Options for ORT_CUDA engine only")
    group.add_argument("--enable-vae-slicing", action="store_true", help="True will feed only one image to VAE once.")
    group.add_argument("--max-cuda-graphs", type=int, default=1, help="Max number of cuda graphs to use. Default 1.")
    group.add_argument("--user-compute-stream", action="store_true", help="Use user compute stream.")

    # TensorRT only options
    group = parser.add_argument_group("Options for TensorRT (--engine=TRT) only")
    group.add_argument(
        "--build-all-tactics", action="store_true", help="Build TensorRT engines using all tactic sources."
    )

    args = parser.parse_args()

    set_default_arguments(args)

    # Validate image dimensions
    if args.height % 64 != 0 or args.width % 64 != 0:
        raise ValueError(
            f"Image height and width have to be divisible by 64 but specified as: {args.height} and {args.width}."
        )

    if (args.build_dynamic_batch or args.build_dynamic_shape) and not args.disable_cuda_graph:
        print("[I] CUDA Graph is disabled since dynamic input shape is configured.")
        args.disable_cuda_graph = True

    if args.onnx_opset is None:
        args.onnx_opset = 14 if args.engine == "ORT_CUDA" else 17

    if is_xl:
        if args.version == "xl-turbo":
            if args.lcm:
                print("[I] sdxl-turbo cannot use with LCM.")
                args.lcm = False

        assert args.strength > 0.0 and args.strength < 1.0

        assert not (args.lcm and args.lora_weights), "it is not supported to use both lcm unet and Lora together"

    if args.scheduler == "LCM":
        if args.guidance > 2.0:
            print("[I] Use --guidance=0.0 (no more than 2.0) when LCM scheduler is used.")
            args.guidance = 0.0
        if args.denoising_steps > 16:
            print("[I] Use --denoising_steps=8 (no more than 16) when LCM scheduler is used.")
            args.denoising_steps = 8

    print(args)

    return args


def max_batch(args):
    if args.max_batch_size:
        max_batch_size = args.max_batch_size
    else:
        do_classifier_free_guidance = args.guidance > 1.0
        batch_multiplier = 2 if do_classifier_free_guidance else 1
        max_batch_size = 32 // batch_multiplier
        if args.engine != "ORT_CUDA" and (args.build_dynamic_shape or args.height > 512 or args.width > 512):
            max_batch_size = 8 // batch_multiplier
    return max_batch_size


def get_metadata(args, is_xl: bool = False) -> dict[str, Any]:
    metadata = {
        "command": " ".join(['"' + x + '"' if " " in x else x for x in sys.argv]),
        "args.prompt": args.prompt,
        "args.negative_prompt": args.negative_prompt,
        "args.batch_size": args.batch_size,
        "height": args.height,
        "width": args.width,
        "cuda_graph": not args.disable_cuda_graph,
        "vae_slicing": args.enable_vae_slicing,
        "engine": args.engine,
    }

    if args.lora_weights:
        metadata["lora_weights"] = args.lora_weights
        metadata["lora_scale"] = args.lora_scale

    if args.controlnet_type:
        metadata["controlnet_type"] = args.controlnet_type
        metadata["controlnet_scale"] = args.controlnet_scale

    if is_xl and args.enable_refiner:
        metadata["base.scheduler"] = args.scheduler
        metadata["base.denoising_steps"] = args.denoising_steps
        metadata["base.guidance"] = args.guidance
        metadata["refiner.strength"] = args.strength
        metadata["refiner.scheduler"] = args.refiner_scheduler
        metadata["refiner.denoising_steps"] = args.refiner_denoising_steps
        metadata["refiner.guidance"] = args.refiner_guidance
    else:
        metadata["scheduler"] = args.scheduler
        metadata["denoising_steps"] = args.denoising_steps
        metadata["guidance"] = args.guidance

    # Version of installed python packages
    packages = ""
    for name in [
        "onnxruntime-gpu",
        "torch",
        "tensorrt",
        "transformers",
        "diffusers",
        "onnx",
        "onnx-graphsurgeon",
        "polygraphy",
        "controlnet_aux",
    ]:
        try:
            packages += (" " if packages else "") + f"{name}=={version(name)}"
        except PackageNotFoundError:
            continue
    metadata["packages"] = packages
    metadata["device"] = torch.cuda.get_device_name()
    metadata["torch.version.cuda"] = torch.version.cuda

    return metadata


def repeat_prompt(args):
    if not isinstance(args.prompt, list):
        raise ValueError(f"`prompt` must be of type `str` or `str` list, but is {type(args.prompt)}")
    prompt = args.prompt * args.batch_size

    if not isinstance(args.negative_prompt, list):
        raise ValueError(
            f"`--negative-prompt` must be of type `str` or `str` list, but is {type(args.negative_prompt)}"
        )

    if len(args.negative_prompt) == 1:
        negative_prompt = args.negative_prompt * len(prompt)
    else:
        negative_prompt = args.negative_prompt

    return prompt, negative_prompt


def initialize_pipeline(
    version="xl-turbo",
    is_refiner: bool = False,
    is_inpaint: bool = False,
    engine_type=EngineType.ORT_CUDA,
    work_dir: str = ".",
    engine_dir=None,
    onnx_opset: int = 17,
    scheduler="EulerA",
    height=512,
    width=512,
    nvtx_profile=False,
    use_cuda_graph=True,
    build_dynamic_batch=False,
    build_dynamic_shape=False,
    min_image_size: int = 512,
    max_image_size: int = 1024,
    max_batch_size: int = 16,
    opt_batch_size: int = 1,
    build_all_tactics: bool = False,
    do_classifier_free_guidance: bool = False,
    lcm: bool = False,
    controlnet=None,
    lora_weights=None,
    lora_scale: float = 1.0,
    use_fp16_vae: bool = True,
    use_vae: bool = True,
    framework_model_dir: str | None = None,
    max_cuda_graphs: int = 1,
):
    pipeline_info = PipelineInfo(
        version,
        is_refiner=is_refiner,
        is_inpaint=is_inpaint,
        use_vae=use_vae,
        min_image_size=min_image_size,
        max_image_size=max_image_size,
        use_fp16_vae=use_fp16_vae,
        use_lcm=lcm,
        do_classifier_free_guidance=do_classifier_free_guidance,
        controlnet=controlnet,
        lora_weights=lora_weights,
        lora_scale=lora_scale,
    )

    input_engine_dir = engine_dir

    onnx_dir, engine_dir, output_dir, framework_model_dir, timing_cache = get_engine_paths(
        work_dir=work_dir, pipeline_info=pipeline_info, engine_type=engine_type, framework_model_dir=framework_model_dir
    )

    pipeline = StableDiffusionPipeline(
        pipeline_info,
        scheduler=scheduler,
        output_dir=output_dir,
        verbose=False,
        nvtx_profile=nvtx_profile,
        max_batch_size=max_batch_size,
        use_cuda_graph=use_cuda_graph,
        framework_model_dir=framework_model_dir,
        engine_type=engine_type,
    )

    import_engine_dir = None
    if input_engine_dir:
        if not os.path.exists(input_engine_dir):
            raise RuntimeError(f"--engine_dir directory does not exist: {input_engine_dir}")

        # Support importing from optimized diffusers onnx pipeline
        if engine_type == EngineType.ORT_CUDA and os.path.exists(os.path.join(input_engine_dir, "model_index.json")):
            import_engine_dir = input_engine_dir
        else:
            engine_dir = input_engine_dir

    opt_image_height = pipeline_info.default_image_size() if build_dynamic_shape else height
    opt_image_width = pipeline_info.default_image_size() if build_dynamic_shape else width

    if engine_type == EngineType.ORT_CUDA:
        pipeline.backend.build_engines(
            engine_dir=engine_dir,
            framework_model_dir=framework_model_dir,
            onnx_dir=onnx_dir,
            tmp_dir=os.path.join(work_dir or ".", engine_type.name, pipeline_info.short_name(), "tmp"),
            device_id=torch.cuda.current_device(),
            import_engine_dir=import_engine_dir,
            max_cuda_graphs=max_cuda_graphs,
        )
    elif engine_type == EngineType.ORT_TRT:
        pipeline.backend.build_engines(
            engine_dir,
            framework_model_dir,
            onnx_dir,
            onnx_opset,
            opt_image_height=opt_image_height,
            opt_image_width=opt_image_width,
            opt_batch_size=opt_batch_size,
            static_batch=not build_dynamic_batch,
            static_image_shape=not build_dynamic_shape,
            max_workspace_size=0,
            device_id=torch.cuda.current_device(),
            timing_cache=timing_cache,
        )
    elif engine_type == EngineType.TRT:
        pipeline.backend.load_engines(
            engine_dir,
            framework_model_dir,
            onnx_dir,
            onnx_opset,
            opt_batch_size=opt_batch_size,
            opt_image_height=opt_image_height,
            opt_image_width=opt_image_width,
            static_batch=not build_dynamic_batch,
            static_shape=not build_dynamic_shape,
            enable_all_tactics=build_all_tactics,
            timing_cache=timing_cache,
        )
    elif engine_type == EngineType.TORCH:
        pipeline.backend.build_engines(framework_model_dir)
    else:
        raise RuntimeError("invalid engine type")

    return pipeline


def load_pipelines(args, batch_size=None):
    engine_type = get_engine_type(args.engine)

    # Register TensorRT plugins
    if engine_type == EngineType.TRT:
        from trt_utilities import init_trt_plugins

        init_trt_plugins()

    max_batch_size = max_batch(args)

    if batch_size is None:
        assert isinstance(args.prompt, list)
        batch_size = len(args.prompt) * args.batch_size

    if batch_size > max_batch_size:
        raise ValueError(f"Batch size {batch_size} is larger than allowed {max_batch_size}.")

    # For TensorRT,  performance of engine built with dynamic shape is very sensitive to the range of image size.
    # Here, we reduce the range of image size for TensorRT to trade-off flexibility and performance.
    # This range can cover most frequent shape of landscape (832x1216), portrait (1216x832) or square (1024x1024).
    if args.version == "xl-turbo":
        min_image_size = 512
        max_image_size = 768 if args.engine != "ORT_CUDA" else 1024
    elif args.version == "xl-1.0":
        min_image_size = 832 if args.engine != "ORT_CUDA" else 512
        max_image_size = 1216 if args.engine != "ORT_CUDA" else 2048
    else:
        # This range can cover common used shape of landscape 512x768, portrait 768x512, or square 512x512 and 768x768.
        min_image_size = 512 if args.engine != "ORT_CUDA" else 256
        max_image_size = 768 if args.engine != "ORT_CUDA" else 1024

    params = {
        "version": args.version,
        "is_refiner": False,
        "is_inpaint": False,
        "engine_type": engine_type,
        "work_dir": args.work_dir,
        "engine_dir": args.engine_dir,
        "onnx_opset": args.onnx_opset,
        "scheduler": args.scheduler,
        "height": args.height,
        "width": args.width,
        "nvtx_profile": args.nvtx_profile,
        "use_cuda_graph": not args.disable_cuda_graph,
        "build_dynamic_batch": args.build_dynamic_batch,
        "build_dynamic_shape": args.build_dynamic_shape,
        "min_image_size": min_image_size,
        "max_image_size": max_image_size,
        "max_batch_size": max_batch_size,
        "opt_batch_size": 1 if args.build_dynamic_batch else batch_size,
        "build_all_tactics": args.build_all_tactics,
        "do_classifier_free_guidance": args.guidance > 1.0,
        "controlnet": args.controlnet_type,
        "lora_weights": args.lora_weights,
        "lora_scale": args.lora_scale,
        "use_fp16_vae": "xl" in args.version,
        "use_vae": True,
        "framework_model_dir": args.framework_model_dir,
        "max_cuda_graphs": args.max_cuda_graphs,
    }

    if "xl" in args.version:
        params["lcm"] = args.lcm
        params["use_vae"] = not args.enable_refiner
    base = initialize_pipeline(**params)

    refiner = None
    if "xl" in args.version and args.enable_refiner:
        params["version"] = "xl-1.0"  # Allow SDXL Turbo to use refiner.
        params["is_refiner"] = True
        params["scheduler"] = args.refiner_scheduler
        params["do_classifier_free_guidance"] = args.refiner_guidance > 1.0
        params["lcm"] = False
        params["controlnet"] = None
        params["lora_weights"] = None
        params["use_vae"] = True
        params["use_fp16_vae"] = True
        refiner = initialize_pipeline(**params)

    if engine_type == EngineType.TRT:
        max_device_memory = max(base.backend.max_device_memory(), (refiner or base).backend.max_device_memory())
        _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
        base.backend.activate_engines(shared_device_memory)
        if refiner:
            refiner.backend.activate_engines(shared_device_memory)

    if engine_type == EngineType.ORT_CUDA:
        enable_vae_slicing = args.enable_vae_slicing
        if batch_size > 4 and not enable_vae_slicing and (args.height >= 1024 and args.width >= 1024):
            print(
                "Updating enable_vae_slicing to be True to avoid cuDNN error for batch size > 4 and resolution >= 1024."
            )
            enable_vae_slicing = True
        if enable_vae_slicing:
            (refiner or base).backend.enable_vae_slicing()
    return base, refiner


def get_depth_image(image):
    """
    Create depth map for SDXL depth control net.
    """
    from transformers import DPTFeatureExtractor, DPTForDepthEstimation

    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    # The depth map is 384x384 by default, here we interpolate to the default output size.
    # Note that it will be resized to output image size later. May change the size here to avoid interpolate twice.
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


def get_canny_image(image) -> Image.Image:
    """
    Create canny image for SDXL control net.
    """
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image


def process_controlnet_images_xl(args) -> list[Image.Image]:
    """
    Process control image for SDXL control net.
    """
    assert len(args.controlnet_image) == 1
    image = Image.open(args.controlnet_image[0]).convert("RGB")

    controlnet_images = []
    if args.controlnet_type[0] == "canny":
        controlnet_images.append(get_canny_image(image))
    elif args.controlnet_type[0] == "depth":
        controlnet_images.append(get_depth_image(image))
    else:
        raise ValueError(f"This controlnet type is not supported for SDXL or Turbo: {args.controlnet_type}.")

    return controlnet_images


def add_controlnet_arguments(parser, is_xl: bool = False):
    """
    Add control net related arguments.
    """
    group = parser.add_argument_group("Options for ControlNet (supports 1.5, sd-turbo, xl-turbo, xl-1.0).")

    group.add_argument(
        "-ci",
        "--controlnet-image",
        nargs="*",
        type=str,
        default=[],
        help="Path to the input regular RGB image/images for controlnet",
    )
    group.add_argument(
        "-ct",
        "--controlnet-type",
        nargs="*",
        type=str,
        default=[],
        choices=list(PipelineInfo.supported_controlnet("xl-1.0" if is_xl else "1.5").keys()),
        help="A list of controlnet type",
    )
    group.add_argument(
        "-cs",
        "--controlnet-scale",
        nargs="*",
        type=float,
        default=[],
        help="The outputs of the controlnet are multiplied by `controlnet_scale` before they are added to the residual in the original unet. Default is 0.5 for SDXL, or 1.0 for SD 1.5",
    )


def process_controlnet_image(controlnet_type: str, image: Image.Image, height, width):
    """
    Process control images of control net v1.1 for Stable Diffusion 1.5.
    """
    control_image = None
    shape = (height, width)
    image = image.convert("RGB")
    if controlnet_type == "canny":
        canny_image = controlnet_aux.CannyDetector()(image)
        control_image = canny_image.resize(shape)
    elif controlnet_type == "normalbae":
        normal_image = controlnet_aux.NormalBaeDetector.from_pretrained("lllyasviel/Annotators")(image)
        control_image = normal_image.resize(shape)
    elif controlnet_type == "depth":
        depth_image = controlnet_aux.LeresDetector.from_pretrained("lllyasviel/Annotators")(image)
        control_image = depth_image.resize(shape)
    elif controlnet_type == "mlsd":
        mlsd_image = controlnet_aux.MLSDdetector.from_pretrained("lllyasviel/Annotators")(image)
        control_image = mlsd_image.resize(shape)
    elif controlnet_type == "openpose":
        openpose_image = controlnet_aux.OpenposeDetector.from_pretrained("lllyasviel/Annotators")(image)
        control_image = openpose_image.resize(shape)
    elif controlnet_type == "scribble":
        scribble_image = controlnet_aux.HEDdetector.from_pretrained("lllyasviel/Annotators")(image, scribble=True)
        control_image = scribble_image.resize(shape)
    elif controlnet_type == "seg":
        seg_image = controlnet_aux.SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")(
            image
        )
        control_image = seg_image.resize(shape)
    else:
        raise ValueError(f"There is no demo image of this controlnet_type: {controlnet_type}")
    return control_image


def process_controlnet_arguments(args):
    """
    Process control net arguments, and returns a list of control images and a tensor of control net scales.
    """
    assert isinstance(args.controlnet_type, list)
    assert isinstance(args.controlnet_scale, list)
    assert isinstance(args.controlnet_image, list)

    if len(args.controlnet_image) != len(args.controlnet_type):
        raise ValueError(
            f"Numbers of controlnet_image {len(args.controlnet_image)} should be equal to number of controlnet_type {len(args.controlnet_type)}."
        )

    if len(args.controlnet_type) == 0:
        return None, None

    if args.version not in ["1.5", "xl-1.0", "xl-turbo", "sd-turbo"]:
        raise ValueError("This demo only supports ControlNet in Stable Diffusion 1.5, XL or Turbo.")

    is_xl = "xl" in args.version
    if is_xl and len(args.controlnet_type) > 1:
        raise ValueError("This demo only support one ControlNet for Stable Diffusion XL or Turbo.")

    if len(args.controlnet_scale) == 0:
        args.controlnet_scale = [0.5 if is_xl else 1.0] * len(args.controlnet_type)
    elif len(args.controlnet_type) != len(args.controlnet_scale):
        raise ValueError(
            f"Numbers of controlnet_type {len(args.controlnet_type)} should be equal to number of controlnet_scale {len(args.controlnet_scale)}."
        )

    # Convert controlnet scales to tensor
    controlnet_scale = torch.FloatTensor(args.controlnet_scale)

    if is_xl:
        images = process_controlnet_images_xl(args)
    else:
        images = []
        for i, image in enumerate(args.controlnet_image):
            images.append(process_controlnet_image(args.controlnet_type[i], Image.open(image), args.height, args.width))

    return images, controlnet_scale

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\humanfriendly\terminal\__init__.py ===
# Human friendly input/output in Python.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: March 1, 2020
# URL: https://humanfriendly.readthedocs.io

"""
Interaction with interactive text terminals.

The :mod:`~humanfriendly.terminal` module makes it easy to interact with
interactive text terminals and format text for rendering on such terminals. If
the terms used in the documentation of this module don't make sense to you then
please refer to the `Wikipedia article on ANSI escape sequences`_ for details
about how ANSI escape sequences work.

This module was originally developed for use on UNIX systems, but since then
Windows 10 gained native support for ANSI escape sequences and this module was
enhanced to recognize and support this. For details please refer to the
:func:`enable_ansi_support()` function.

.. _Wikipedia article on ANSI escape sequences: http://en.wikipedia.org/wiki/ANSI_escape_code#Sequence_elements
"""

# Standard library modules.
import codecs
import numbers
import os
import platform
import re
import subprocess
import sys

# The `fcntl' module is platform specific so importing it may give an error. We
# hide this implementation detail from callers by handling the import error and
# setting a flag instead.
try:
    import fcntl
    import termios
    import struct
    HAVE_IOCTL = True
except ImportError:
    HAVE_IOCTL = False

# Modules included in our package.
from humanfriendly.compat import coerce_string, is_unicode, on_windows, which
from humanfriendly.decorators import cached
from humanfriendly.deprecation import define_aliases
from humanfriendly.text import concatenate, format
from humanfriendly.usage import format_usage

# Public identifiers that require documentation.
__all__ = (
    'ANSI_COLOR_CODES',
    'ANSI_CSI',
    'ANSI_ERASE_LINE',
    'ANSI_HIDE_CURSOR',
    'ANSI_RESET',
    'ANSI_SGR',
    'ANSI_SHOW_CURSOR',
    'ANSI_TEXT_STYLES',
    'CLEAN_OUTPUT_PATTERN',
    'DEFAULT_COLUMNS',
    'DEFAULT_ENCODING',
    'DEFAULT_LINES',
    'HIGHLIGHT_COLOR',
    'ansi_strip',
    'ansi_style',
    'ansi_width',
    'ansi_wrap',
    'auto_encode',
    'clean_terminal_output',
    'connected_to_terminal',
    'enable_ansi_support',
    'find_terminal_size',
    'find_terminal_size_using_ioctl',
    'find_terminal_size_using_stty',
    'get_pager_command',
    'have_windows_native_ansi_support',
    'message',
    'output',
    'readline_strip',
    'readline_wrap',
    'show_pager',
    'terminal_supports_colors',
    'usage',
    'warning',
)

ANSI_CSI = '\x1b['
"""The ANSI "Control Sequence Introducer" (a string)."""

ANSI_SGR = 'm'
"""The ANSI "Select Graphic Rendition" sequence (a string)."""

ANSI_ERASE_LINE = '%sK' % ANSI_CSI
"""The ANSI escape sequence to erase the current line (a string)."""

ANSI_RESET = '%s0%s' % (ANSI_CSI, ANSI_SGR)
"""The ANSI escape sequence to reset styling (a string)."""

ANSI_HIDE_CURSOR = '%s?25l' % ANSI_CSI
"""The ANSI escape sequence to hide the text cursor (a string)."""

ANSI_SHOW_CURSOR = '%s?25h' % ANSI_CSI
"""The ANSI escape sequence to show the text cursor (a string)."""

ANSI_COLOR_CODES = dict(black=0, red=1, green=2, yellow=3, blue=4, magenta=5, cyan=6, white=7)
"""
A dictionary with (name, number) pairs of `portable color codes`_. Used by
:func:`ansi_style()` to generate ANSI escape sequences that change font color.

.. _portable color codes: http://en.wikipedia.org/wiki/ANSI_escape_code#Colors
"""

ANSI_TEXT_STYLES = dict(bold=1, faint=2, italic=3, underline=4, inverse=7, strike_through=9)
"""
A dictionary with (name, number) pairs of text styles (effects). Used by
:func:`ansi_style()` to generate ANSI escape sequences that change text
styles. Only widely supported text styles are included here.
"""

CLEAN_OUTPUT_PATTERN = re.compile(u'(\r|\n|\b|%s)' % re.escape(ANSI_ERASE_LINE))
"""
A compiled regular expression used to separate significant characters from other text.

This pattern is used by :func:`clean_terminal_output()` to split terminal
output into regular text versus backspace, carriage return and line feed
characters and ANSI 'erase line' escape sequences.
"""

DEFAULT_LINES = 25
"""The default number of lines in a terminal (an integer)."""

DEFAULT_COLUMNS = 80
"""The default number of columns in a terminal (an integer)."""

DEFAULT_ENCODING = 'UTF-8'
"""The output encoding for Unicode strings."""

HIGHLIGHT_COLOR = os.environ.get('HUMANFRIENDLY_HIGHLIGHT_COLOR', 'green')
"""
The color used to highlight important tokens in formatted text (e.g. the usage
message of the ``humanfriendly`` program). If the environment variable
``$HUMANFRIENDLY_HIGHLIGHT_COLOR`` is set it determines the value of
:data:`HIGHLIGHT_COLOR`.
"""


def ansi_strip(text, readline_hints=True):
    """
    Strip ANSI escape sequences from the given string.

    :param text: The text from which ANSI escape sequences should be removed (a
                 string).
    :param readline_hints: If :data:`True` then :func:`readline_strip()` is
                           used to remove `readline hints`_ from the string.
    :returns: The text without ANSI escape sequences (a string).
    """
    pattern = '%s.*?%s' % (re.escape(ANSI_CSI), re.escape(ANSI_SGR))
    text = re.sub(pattern, '', text)
    if readline_hints:
        text = readline_strip(text)
    return text


def ansi_style(**kw):
    """
    Generate ANSI escape sequences for the given color and/or style(s).

    :param color: The foreground color. Three types of values are supported:

                  - The name of a color (one of the strings 'black', 'red',
                    'green', 'yellow', 'blue', 'magenta', 'cyan' or 'white').
                  - An integer that refers to the 256 color mode palette.
                  - A tuple or list with three integers representing an RGB
                    (red, green, blue) value.

                  The value :data:`None` (the default) means no escape
                  sequence to switch color will be emitted.
    :param background: The background color (see the description
                       of the `color` argument).
    :param bright: Use high intensity colors instead of default colors
                   (a boolean, defaults to :data:`False`).
    :param readline_hints: If :data:`True` then :func:`readline_wrap()` is
                           applied to the generated ANSI escape sequences (the
                           default is :data:`False`).
    :param kw: Any additional keyword arguments are expected to match a key
               in the :data:`ANSI_TEXT_STYLES` dictionary. If the argument's
               value evaluates to :data:`True` the respective style will be
               enabled.
    :returns: The ANSI escape sequences to enable the requested text styles or
              an empty string if no styles were requested.
    :raises: :exc:`~exceptions.ValueError` when an invalid color name is given.

    Even though only eight named colors are supported, the use of `bright=True`
    and `faint=True` increases the number of available colors to around 24 (it
    may be slightly lower, for example because faint black is just black).

    **Support for 8-bit colors**

    In `release 4.7`_ support for 256 color mode was added. While this
    significantly increases the available colors it's not very human friendly
    in usage because you need to look up color codes in the `256 color mode
    palette <https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit>`_.

    You can use the ``humanfriendly --demo`` command to get a demonstration of
    the available colors, see also the screen shot below. Note that the small
    font size in the screen shot was so that the demonstration of 256 color
    mode support would fit into a single screen shot without scrolling :-)
    (I wasn't feeling very creative).

      .. image:: images/ansi-demo.png

    **Support for 24-bit colors**

    In `release 4.14`_ support for 24-bit colors was added by accepting a tuple
    or list with three integers representing the RGB (red, green, blue) value
    of a color. This is not included in the demo because rendering millions of
    colors was deemed unpractical ;-).

    .. _release 4.7: http://humanfriendly.readthedocs.io/en/latest/changelog.html#release-4-7-2018-01-14
    .. _release 4.14: http://humanfriendly.readthedocs.io/en/latest/changelog.html#release-4-14-2018-07-13
    """
    # Start with sequences that change text styles.
    sequences = [ANSI_TEXT_STYLES[k] for k, v in kw.items() if k in ANSI_TEXT_STYLES and v]
    # Append the color code (if any).
    for color_type in 'color', 'background':
        color_value = kw.get(color_type)
        if isinstance(color_value, (tuple, list)):
            if len(color_value) != 3:
                msg = "Invalid color value %r! (expected tuple or list with three numbers)"
                raise ValueError(msg % color_value)
            sequences.append(48 if color_type == 'background' else 38)
            sequences.append(2)
            sequences.extend(map(int, color_value))
        elif isinstance(color_value, numbers.Number):
            # Numeric values are assumed to be 256 color codes.
            sequences.extend((
                39 if color_type == 'background' else 38,
                5, int(color_value)
            ))
        elif color_value:
            # Other values are assumed to be strings containing one of the known color names.
            if color_value not in ANSI_COLOR_CODES:
                msg = "Invalid color value %r! (expected an integer or one of the strings %s)"
                raise ValueError(msg % (color_value, concatenate(map(repr, sorted(ANSI_COLOR_CODES)))))
            # Pick the right offset for foreground versus background
            # colors and regular intensity versus bright colors.
            offset = (
                (100 if kw.get('bright') else 40)
                if color_type == 'background'
                else (90 if kw.get('bright') else 30)
            )
            # Combine the offset and color code into a single integer.
            sequences.append(offset + ANSI_COLOR_CODES[color_value])
    if sequences:
        encoded = ANSI_CSI + ';'.join(map(str, sequences)) + ANSI_SGR
        return readline_wrap(encoded) if kw.get('readline_hints') else encoded
    else:
        return ''


def ansi_width(text):
    """
    Calculate the effective width of the given text (ignoring ANSI escape sequences).

    :param text: The text whose width should be calculated (a string).
    :returns: The width of the text without ANSI escape sequences (an
              integer).

    This function uses :func:`ansi_strip()` to strip ANSI escape sequences from
    the given string and returns the length of the resulting string.
    """
    return len(ansi_strip(text))


def ansi_wrap(text, **kw):
    """
    Wrap text in ANSI escape sequences for the given color and/or style(s).

    :param text: The text to wrap (a string).
    :param kw: Any keyword arguments are passed to :func:`ansi_style()`.
    :returns: The result of this function depends on the keyword arguments:

              - If :func:`ansi_style()` generates an ANSI escape sequence based
                on the keyword arguments, the given text is prefixed with the
                generated ANSI escape sequence and suffixed with
                :data:`ANSI_RESET`.

              - If :func:`ansi_style()` returns an empty string then the text
                given by the caller is returned unchanged.
    """
    start_sequence = ansi_style(**kw)
    if start_sequence:
        end_sequence = ANSI_RESET
        if kw.get('readline_hints'):
            end_sequence = readline_wrap(end_sequence)
        return start_sequence + text + end_sequence
    else:
        return text


def auto_encode(stream, text, *args, **kw):
    """
    Reliably write Unicode strings to the terminal.

    :param stream: The file-like object to write to (a value like
                   :data:`sys.stdout` or :data:`sys.stderr`).
    :param text: The text to write to the stream (a string).
    :param args: Refer to :func:`~humanfriendly.text.format()`.
    :param kw: Refer to :func:`~humanfriendly.text.format()`.

    Renders the text using :func:`~humanfriendly.text.format()` and writes it
    to the given stream. If an :exc:`~exceptions.UnicodeEncodeError` is
    encountered in doing so, the text is encoded using :data:`DEFAULT_ENCODING`
    and the write is retried. The reasoning behind this rather blunt approach
    is that it's preferable to get output on the command line in the wrong
    encoding then to have the Python program blow up with a
    :exc:`~exceptions.UnicodeEncodeError` exception.
    """
    text = format(text, *args, **kw)
    try:
        stream.write(text)
    except UnicodeEncodeError:
        stream.write(codecs.encode(text, DEFAULT_ENCODING))


def clean_terminal_output(text):
    """
    Clean up the terminal output of a command.

    :param text: The raw text with special characters (a Unicode string).
    :returns: A list of Unicode strings (one for each line).

    This function emulates the effect of backspace (0x08), carriage return
    (0x0D) and line feed (0x0A) characters and the ANSI 'erase line' escape
    sequence on interactive terminals. It's intended to clean up command output
    that was originally meant to be rendered on an interactive terminal and
    that has been captured using e.g. the :man:`script` program [#]_ or the
    :mod:`pty` module [#]_.

    .. [#] My coloredlogs_ package supports the ``coloredlogs --to-html``
           command which uses :man:`script` to fool a subprocess into thinking
           that it's connected to an interactive terminal (in order to get it
           to emit ANSI escape sequences).

    .. [#] My capturer_ package uses the :mod:`pty` module to fool the current
           process and subprocesses into thinking they are connected to an
           interactive terminal (in order to get them to emit ANSI escape
           sequences).

    **Some caveats about the use of this function:**

    - Strictly speaking the effect of carriage returns cannot be emulated
      outside of an actual terminal due to the interaction between overlapping
      output, terminal widths and line wrapping. The goal of this function is
      to sanitize noise in terminal output while preserving useful output.
      Think of it as a useful and pragmatic but possibly lossy conversion.

    - The algorithm isn't smart enough to properly handle a pair of ANSI escape
      sequences that open before a carriage return and close after the last
      carriage return in a linefeed delimited string; the resulting string will
      contain only the closing end of the ANSI escape sequence pair. Tracking
      this kind of complexity requires a state machine and proper parsing.

    .. _capturer: https://pypi.org/project/capturer
    .. _coloredlogs: https://pypi.org/project/coloredlogs
    """
    cleaned_lines = []
    current_line = ''
    current_position = 0
    for token in CLEAN_OUTPUT_PATTERN.split(text):
        if token == '\r':
            # Seek back to the start of the current line.
            current_position = 0
        elif token == '\b':
            # Seek back one character in the current line.
            current_position = max(0, current_position - 1)
        else:
            if token == '\n':
                # Capture the current line.
                cleaned_lines.append(current_line)
            if token in ('\n', ANSI_ERASE_LINE):
                # Clear the current line.
                current_line = ''
                current_position = 0
            elif token:
                # Merge regular output into the current line.
                new_position = current_position + len(token)
                prefix = current_line[:current_position]
                suffix = current_line[new_position:]
                current_line = prefix + token + suffix
                current_position = new_position
    # Capture the last line (if any).
    cleaned_lines.append(current_line)
    # Remove any empty trailing lines.
    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop(-1)
    return cleaned_lines


def connected_to_terminal(stream=None):
    """
    Check if a stream is connected to a terminal.

    :param stream: The stream to check (a file-like object,
                   defaults to :data:`sys.stdout`).
    :returns: :data:`True` if the stream is connected to a terminal,
              :data:`False` otherwise.

    See also :func:`terminal_supports_colors()`.
    """
    stream = sys.stdout if stream is None else stream
    try:
        return stream.isatty()
    except Exception:
        return False


@cached
def enable_ansi_support():
    """
    Try to enable support for ANSI escape sequences (required on Windows).

    :returns: :data:`True` if ANSI is supported, :data:`False` otherwise.

    This functions checks for the following supported configurations, in the
    given order:

    1. On Windows, if :func:`have_windows_native_ansi_support()` confirms
       native support for ANSI escape sequences :mod:`ctypes` will be used to
       enable this support.

    2. On Windows, if the environment variable ``$ANSICON`` is set nothing is
       done because it is assumed that support for ANSI escape sequences has
       already been enabled via `ansicon <https://github.com/adoxa/ansicon>`_.

    3. On Windows, an attempt is made to import and initialize the Python
       package :pypi:`colorama` instead (of course for this to work
       :pypi:`colorama` has to be installed).

    4. On other platforms this function calls :func:`connected_to_terminal()`
       to determine whether ANSI escape sequences are supported (that is to
       say all platforms that are not Windows are assumed to support ANSI
       escape sequences natively, without weird contortions like above).

       This makes it possible to call :func:`enable_ansi_support()`
       unconditionally without checking the current platform.

    The :func:`~humanfriendly.decorators.cached` decorator is used to ensure
    that this function is only executed once, but its return value remains
    available on later calls.
    """
    if have_windows_native_ansi_support():
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
        ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-12), 7)
        return True
    elif on_windows():
        if 'ANSICON' in os.environ:
            return True
        try:
            import colorama
            colorama.init()
            return True
        except ImportError:
            return False
    else:
        return connected_to_terminal()


def find_terminal_size():
    """
    Determine the number of lines and columns visible in the terminal.

    :returns: A tuple of two integers with the line and column count.

    The result of this function is based on the first of the following three
    methods that works:

    1. First :func:`find_terminal_size_using_ioctl()` is tried,
    2. then :func:`find_terminal_size_using_stty()` is tried,
    3. finally :data:`DEFAULT_LINES` and :data:`DEFAULT_COLUMNS` are returned.

    .. note:: The :func:`find_terminal_size()` function performs the steps
              above every time it is called, the result is not cached. This is
              because the size of a virtual terminal can change at any time and
              the result of :func:`find_terminal_size()` should be correct.

              `Pre-emptive snarky comment`_: It's possible to cache the result
              of this function and use :mod:`signal.SIGWINCH <signal>` to
              refresh the cached values!

              Response: As a library I don't consider it the role of the
              :mod:`humanfriendly.terminal` module to install a process wide
              signal handler ...

    .. _Pre-emptive snarky comment: http://blogs.msdn.com/b/oldnewthing/archive/2008/01/30/7315957.aspx
    """
    # The first method. Any of the standard streams may have been redirected
    # somewhere and there's no telling which, so we'll just try them all.
    for stream in sys.stdin, sys.stdout, sys.stderr:
        try:
            result = find_terminal_size_using_ioctl(stream)
            if min(result) >= 1:
                return result
        except Exception:
            pass
    # The second method.
    try:
        result = find_terminal_size_using_stty()
        if min(result) >= 1:
            return result
    except Exception:
        pass
    # Fall back to conservative defaults.
    return DEFAULT_LINES, DEFAULT_COLUMNS


def find_terminal_size_using_ioctl(stream):
    """
    Find the terminal size using :func:`fcntl.ioctl()`.

    :param stream: A stream connected to the terminal (a file object with a
                   ``fileno`` attribute).
    :returns: A tuple of two integers with the line and column count.
    :raises: This function can raise exceptions but I'm not going to document
             them here, you should be using :func:`find_terminal_size()`.

    Based on an `implementation found on StackOverflow <http://stackoverflow.com/a/3010495/788200>`_.
    """
    if not HAVE_IOCTL:
        raise NotImplementedError("It looks like the `fcntl' module is not available!")
    h, w, hp, wp = struct.unpack('HHHH', fcntl.ioctl(stream, termios.TIOCGWINSZ, struct.pack('HHHH', 0, 0, 0, 0)))
    return h, w


def find_terminal_size_using_stty():
    """
    Find the terminal size using the external command ``stty size``.

    :param stream: A stream connected to the terminal (a file object).
    :returns: A tuple of two integers with the line and column count.
    :raises: This function can raise exceptions but I'm not going to document
             them here, you should be using :func:`find_terminal_size()`.
    """
    stty = subprocess.Popen(['stty', 'size'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    stdout, stderr = stty.communicate()
    tokens = stdout.split()
    if len(tokens) != 2:
        raise Exception("Invalid output from `stty size'!")
    return tuple(map(int, tokens))


def get_pager_command(text=None):
    """
    Get the command to show a text on the terminal using a pager.

    :param text: The text to print to the terminal (a string).
    :returns: A list of strings with the pager command and arguments.

    The use of a pager helps to avoid the wall of text effect where the user
    has to scroll up to see where the output began (not very user friendly).

    If the given text contains ANSI escape sequences the command ``less
    --RAW-CONTROL-CHARS`` is used, otherwise the environment variable
    ``$PAGER`` is used (if ``$PAGER`` isn't set :man:`less` is used).

    When the selected pager is :man:`less`, the following options are used to
    make the experience more user friendly:

    - ``--quit-if-one-screen`` causes :man:`less` to automatically exit if the
      entire text can be displayed on the first screen. This makes the use of a
      pager transparent for smaller texts (because the operator doesn't have to
      quit the pager).

    - ``--no-init`` prevents :man:`less` from clearing the screen when it
      exits. This ensures that the operator gets a chance to review the text
      (for example a usage message) after quitting the pager, while composing
      the next command.
    """
    # Compose the pager command.
    if text and ANSI_CSI in text:
        command_line = ['less', '--RAW-CONTROL-CHARS']
    else:
        command_line = [os.environ.get('PAGER', 'less')]
    # Pass some additional options to `less' (to make it more
    # user friendly) without breaking support for other pagers.
    if os.path.basename(command_line[0]) == 'less':
        command_line.append('--no-init')
        command_line.append('--quit-if-one-screen')
    return command_line


@cached
def have_windows_native_ansi_support():
    """
    Check if we're running on a Windows 10 release with native support for ANSI escape sequences.

    :returns: :data:`True` if so, :data:`False` otherwise.

    The :func:`~humanfriendly.decorators.cached` decorator is used as a minor
    performance optimization. Semantically this should have zero impact because
    the answer doesn't change in the lifetime of a computer process.
    """
    if on_windows():
        try:
            # I can't be 100% sure this will never break and I'm not in a
            # position to test it thoroughly either, so I decided that paying
            # the price of one additional try / except statement is worth the
            # additional peace of mind :-).
            components = tuple(int(c) for c in platform.version().split('.'))
            return components >= (10, 0, 14393)
        except Exception:
            pass
    return False


def message(text, *args, **kw):
    """
    Print a formatted message to the standard error stream.

    For details about argument handling please refer to
    :func:`~humanfriendly.text.format()`.

    Renders the message using :func:`~humanfriendly.text.format()` and writes
    the resulting string (followed by a newline) to :data:`sys.stderr` using
    :func:`auto_encode()`.
    """
    auto_encode(sys.stderr, coerce_string(text) + '\n', *args, **kw)


def output(text, *args, **kw):
    """
    Print a formatted message to the standard output stream.

    For details about argument handling please refer to
    :func:`~humanfriendly.text.format()`.

    Renders the message using :func:`~humanfriendly.text.format()` and writes
    the resulting string (followed by a newline) to :data:`sys.stdout` using
    :func:`auto_encode()`.
    """
    auto_encode(sys.stdout, coerce_string(text) + '\n', *args, **kw)


def readline_strip(expr):
    """
    Remove `readline hints`_ from a string.

    :param text: The text to strip (a string).
    :returns: The stripped text.
    """
    return expr.replace('\001', '').replace('\002', '')


def readline_wrap(expr):
    """
    Wrap an ANSI escape sequence in `readline hints`_.

    :param text: The text with the escape sequence to wrap (a string).
    :returns: The wrapped text.

    .. _readline hints: http://superuser.com/a/301355
    """
    return '\001' + expr + '\002'


def show_pager(formatted_text, encoding=DEFAULT_ENCODING):
    """
    Print a large text to the terminal using a pager.

    :param formatted_text: The text to print to the terminal (a string).
    :param encoding: The name of the text encoding used to encode the formatted
                     text if the formatted text is a Unicode string (a string,
                     defaults to :data:`DEFAULT_ENCODING`).

    When :func:`connected_to_terminal()` returns :data:`True` a pager is used
    to show the text on the terminal, otherwise the text is printed directly
    without invoking a pager.

    The use of a pager helps to avoid the wall of text effect where the user
    has to scroll up to see where the output began (not very user friendly).

    Refer to :func:`get_pager_command()` for details about the command line
    that's used to invoke the pager.
    """
    if connected_to_terminal():
        # Make sure the selected pager command is available.
        command_line = get_pager_command(formatted_text)
        if which(command_line[0]):
            pager = subprocess.Popen(command_line, stdin=subprocess.PIPE)
            if is_unicode(formatted_text):
                formatted_text = formatted_text.encode(encoding)
            pager.communicate(input=formatted_text)
            return
    output(formatted_text)


def terminal_supports_colors(stream=None):
    """
    Check if a stream is connected to a terminal that supports ANSI escape sequences.

    :param stream: The stream to check (a file-like object,
                   defaults to :data:`sys.stdout`).
    :returns: :data:`True` if the terminal supports ANSI escape sequences,
              :data:`False` otherwise.

    This function was originally inspired by the implementation of
    `django.core.management.color.supports_color()
    <https://github.com/django/django/blob/master/django/core/management/color.py>`_
    but has since evolved significantly.
    """
    if on_windows():
        # On Windows support for ANSI escape sequences is not a given.
        have_ansicon = 'ANSICON' in os.environ
        have_colorama = 'colorama' in sys.modules
        have_native_support = have_windows_native_ansi_support()
        if not (have_ansicon or have_colorama or have_native_support):
            return False
    return connected_to_terminal(stream)


def usage(usage_text):
    """
    Print a human friendly usage message to the terminal.

    :param text: The usage message to print (a string).

    This function does two things:

    1. If :data:`sys.stdout` is connected to a terminal (see
       :func:`connected_to_terminal()`) then the usage message is formatted
       using :func:`.format_usage()`.
    2. The usage message is shown using a pager (see :func:`show_pager()`).
    """
    if terminal_supports_colors(sys.stdout):
        usage_text = format_usage(usage_text)
    show_pager(usage_text)


def warning(text, *args, **kw):
    """
    Show a warning message on the terminal.

    For details about argument handling please refer to
    :func:`~humanfriendly.text.format()`.

    Renders the message using :func:`~humanfriendly.text.format()` and writes
    the resulting string (followed by a newline) to :data:`sys.stderr` using
    :func:`auto_encode()`.

    If :data:`sys.stderr` is connected to a terminal that supports colors,
    :func:`ansi_wrap()` is used to color the message in a red font (to make
    the warning stand out from surrounding text).
    """
    text = coerce_string(text)
    if terminal_supports_colors(sys.stderr):
        text = ansi_wrap(text, color='red')
    auto_encode(sys.stderr, text + '\n', *args, **kw)


# Define aliases for backwards compatibility.
define_aliases(
    module_name=__name__,
    # In humanfriendly 1.31 the find_meta_variables() and format_usage()
    # functions were extracted to the new module humanfriendly.usage.
    find_meta_variables='humanfriendly.usage.find_meta_variables',
    format_usage='humanfriendly.usage.format_usage',
    # In humanfriendly 8.0 the html_to_ansi() function and HTMLConverter
    # class were extracted to the new module humanfriendly.terminal.html.
    html_to_ansi='humanfriendly.terminal.html.html_to_ansi',
    HTMLConverter='humanfriendly.terminal.html.HTMLConverter',
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\browser.py ===
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
    LOCAL_NETWORK_ACCESS = "localNetworkAccess"
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
    OPEN_GLIC = "openGlic"

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


class PrivacySandboxAPI(enum.Enum):
    BIDDING_AND_AUCTION_SERVICES = "BiddingAndAuctionServices"
    TRUSTED_KEY_VALUE = "TrustedKeyValue"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


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


def add_privacy_sandbox_coordinator_key_config(
        api: PrivacySandboxAPI,
        coordinator_origin: str,
        key_config: str,
        browser_context_id: typing.Optional[BrowserContextID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Configures encryption keys used with a given privacy sandbox API to talk
    to a trusted coordinator.  Since this is intended for test automation only,
    coordinatorOrigin must be a .test domain. No existing coordinator
    configuration for the origin may exist.

    :param api:
    :param coordinator_origin:
    :param key_config:
    :param browser_context_id: *(Optional)* BrowserContext to perform the action in. When omitted, default browser context is used.
    '''
    params: T_JSON_DICT = dict()
    params['api'] = api.to_json()
    params['coordinatorOrigin'] = coordinator_origin
    params['keyConfig'] = key_config
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.addPrivacySandboxCoordinatorKeyConfig',
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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\browser.py ===
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
    LOCAL_NETWORK_ACCESS = "localNetworkAccess"
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


class PrivacySandboxAPI(enum.Enum):
    BIDDING_AND_AUCTION_SERVICES = "BiddingAndAuctionServices"
    TRUSTED_KEY_VALUE = "TrustedKeyValue"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


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


def add_privacy_sandbox_coordinator_key_config(
        api: PrivacySandboxAPI,
        coordinator_origin: str,
        key_config: str,
        browser_context_id: typing.Optional[BrowserContextID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Configures encryption keys used with a given privacy sandbox API to talk
    to a trusted coordinator.  Since this is intended for test automation only,
    coordinatorOrigin must be a .test domain. No existing coordinator
    configuration for the origin may exist.

    :param api:
    :param coordinator_origin:
    :param key_config:
    :param browser_context_id: *(Optional)* BrowserContext to perform the action in. When omitted, default browser context is used.
    '''
    params: T_JSON_DICT = dict()
    params['api'] = api.to_json()
    params['coordinatorOrigin'] = coordinator_origin
    params['keyConfig'] = key_config
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.addPrivacySandboxCoordinatorKeyConfig',
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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\ma\mrecords.py ===
""":mod:`numpy.ma..mrecords`

Defines the equivalent of :class:`numpy.recarrays` for masked arrays,
where fields can be accessed as attributes.
Note that :class:`numpy.ma.MaskedArray` already supports structured datatypes
and the masking of individual fields.

.. moduleauthor:: Pierre Gerard-Marchant

"""
#  We should make sure that no field is called '_mask','mask','_fieldmask',
#  or whatever restricted keywords.  An idea would be to no bother in the
#  first place, and then rename the invalid fields with a trailing
#  underscore. Maybe we could just overload the parser function ?

import warnings

import numpy as np
import numpy.ma as ma

_byteorderconv = np._core.records._byteorderconv


_check_fill_value = ma.core._check_fill_value


__all__ = [
    'MaskedRecords', 'mrecarray', 'fromarrays', 'fromrecords',
    'fromtextfile', 'addfield',
]

reserved_fields = ['_data', '_mask', '_fieldmask', 'dtype']


def _checknames(descr, names=None):
    """
    Checks that field names ``descr`` are not reserved keywords.

    If this is the case, a default 'f%i' is substituted.  If the argument
    `names` is not None, updates the field names to valid names.

    """
    ndescr = len(descr)
    default_names = [f'f{i}' for i in range(ndescr)]
    if names is None:
        new_names = default_names
    else:
        if isinstance(names, (tuple, list)):
            new_names = names
        elif isinstance(names, str):
            new_names = names.split(',')
        else:
            raise NameError(f'illegal input names {names!r}')
        nnames = len(new_names)
        if nnames < ndescr:
            new_names += default_names[nnames:]
    ndescr = []
    for (n, d, t) in zip(new_names, default_names, descr.descr):
        if n in reserved_fields:
            if t[0] in reserved_fields:
                ndescr.append((d, t[1]))
            else:
                ndescr.append(t)
        else:
            ndescr.append((n, t[1]))
    return np.dtype(ndescr)


def _get_fieldmask(self):
    mdescr = [(n, '|b1') for n in self.dtype.names]
    fdmask = np.empty(self.shape, dtype=mdescr)
    fdmask.flat = tuple([False] * len(mdescr))
    return fdmask


class MaskedRecords(ma.MaskedArray):
    """

    Attributes
    ----------
    _data : recarray
        Underlying data, as a record array.
    _mask : boolean array
        Mask of the records. A record is masked when all its fields are
        masked.
    _fieldmask : boolean recarray
        Record array of booleans, setting the mask of each individual field
        of each record.
    _fill_value : record
        Filling values for each field.

    """

    def __new__(cls, shape, dtype=None, buf=None, offset=0, strides=None,
                formats=None, names=None, titles=None,
                byteorder=None, aligned=False,
                mask=ma.nomask, hard_mask=False, fill_value=None, keep_mask=True,
                copy=False,
                **options):

        self = np.recarray.__new__(cls, shape, dtype=dtype, buf=buf, offset=offset,
                                   strides=strides, formats=formats, names=names,
                                   titles=titles, byteorder=byteorder,
                                   aligned=aligned,)

        mdtype = ma.make_mask_descr(self.dtype)
        if mask is ma.nomask or not np.size(mask):
            if not keep_mask:
                self._mask = tuple([False] * len(mdtype))
        else:
            mask = np.array(mask, copy=copy)
            if mask.shape != self.shape:
                (nd, nm) = (self.size, mask.size)
                if nm == 1:
                    mask = np.resize(mask, self.shape)
                elif nm == nd:
                    mask = np.reshape(mask, self.shape)
                else:
                    msg = (f"Mask and data not compatible: data size is {nd},"
                           " mask size is {nm}.")
                    raise ma.MAError(msg)
            if not keep_mask:
                self.__setmask__(mask)
                self._sharedmask = True
            else:
                if mask.dtype == mdtype:
                    _mask = mask
                else:
                    _mask = np.array([tuple([m] * len(mdtype)) for m in mask],
                                     dtype=mdtype)
                self._mask = _mask
        return self

    def __array_finalize__(self, obj):
        # Make sure we have a _fieldmask by default
        _mask = getattr(obj, '_mask', None)
        if _mask is None:
            objmask = getattr(obj, '_mask', ma.nomask)
            _dtype = np.ndarray.__getattribute__(self, 'dtype')
            if objmask is ma.nomask:
                _mask = ma.make_mask_none(self.shape, dtype=_dtype)
            else:
                mdescr = ma.make_mask_descr(_dtype)
                _mask = np.array([tuple([m] * len(mdescr)) for m in objmask],
                               dtype=mdescr).view(np.recarray)
        # Update some of the attributes
        _dict = self.__dict__
        _dict.update(_mask=_mask)
        self._update_from(obj)
        if _dict['_baseclass'] == np.ndarray:
            _dict['_baseclass'] = np.recarray

    @property
    def _data(self):
        """
        Returns the data as a recarray.

        """
        return np.ndarray.view(self, np.recarray)

    @property
    def _fieldmask(self):
        """
        Alias to mask.

        """
        return self._mask

    def __len__(self):
        """
        Returns the length

        """
        # We have more than one record
        if self.ndim:
            return len(self._data)
        # We have only one record: return the nb of fields
        return len(self.dtype)

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            # attr must be a fieldname
            pass
        fielddict = np.ndarray.__getattribute__(self, 'dtype').fields
        try:
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            raise AttributeError(
                f'record array has no attribute {attr}') from e
        # So far, so good
        _localdict = np.ndarray.__getattribute__(self, '__dict__')
        _data = np.ndarray.view(self, _localdict['_baseclass'])
        obj = _data.getfield(*res)
        if obj.dtype.names is not None:
            raise NotImplementedError("MaskedRecords is currently limited to"
                                      "simple records.")
        # Get some special attributes
        # Reset the object's mask
        hasmasked = False
        _mask = _localdict.get('_mask', None)
        if _mask is not None:
            try:
                _mask = _mask[attr]
            except IndexError:
                # Couldn't find a mask: use the default (nomask)
                pass
            tp_len = len(_mask.dtype)
            hasmasked = _mask.view((bool, ((tp_len,) if tp_len else ()))).any()
        if (obj.shape or hasmasked):
            obj = obj.view(ma.MaskedArray)
            obj._baseclass = np.ndarray
            obj._isfield = True
            obj._mask = _mask
            # Reset the field values
            _fill_value = _localdict.get('_fill_value', None)
            if _fill_value is not None:
                try:
                    obj._fill_value = _fill_value[attr]
                except ValueError:
                    obj._fill_value = None
        else:
            obj = obj.item()
        return obj

    def __setattr__(self, attr, val):
        """
        Sets the attribute attr to the value val.

        """
        # Should we call __setmask__ first ?
        if attr in ['mask', 'fieldmask']:
            self.__setmask__(val)
            return
        # Create a shortcut (so that we don't have to call getattr all the time)
        _localdict = object.__getattribute__(self, '__dict__')
        # Check whether we're creating a new field
        newattr = attr not in _localdict
        try:
            # Is attr a generic attribute ?
            ret = object.__setattr__(self, attr, val)
        except Exception:
            # Not a generic attribute: exit if it's not a valid field
            fielddict = np.ndarray.__getattribute__(self, 'dtype').fields or {}
            optinfo = np.ndarray.__getattribute__(self, '_optinfo') or {}
            if not (attr in fielddict or attr in optinfo):
                raise
        else:
            # Get the list of names
            fielddict = np.ndarray.__getattribute__(self, 'dtype').fields or {}
            # Check the attribute
            if attr not in fielddict:
                return ret
            if newattr:
                # We just added this one or this setattr worked on an
                # internal attribute.
                try:
                    object.__delattr__(self, attr)
                except Exception:
                    return ret
        # Let's try to set the field
        try:
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            raise AttributeError(
                f'record array has no attribute {attr}') from e

        if val is ma.masked:
            _fill_value = _localdict['_fill_value']
            if _fill_value is not None:
                dval = _localdict['_fill_value'][attr]
            else:
                dval = val
            mval = True
        else:
            dval = ma.filled(val)
            mval = ma.getmaskarray(val)
        obj = np.ndarray.__getattribute__(self, '_data').setfield(dval, *res)
        _localdict['_mask'].__setitem__(attr, mval)
        return obj

    def __getitem__(self, indx):
        """
        Returns all the fields sharing the same fieldname base.

        The fieldname base is either `_data` or `_mask`.

        """
        _localdict = self.__dict__
        _mask = np.ndarray.__getattribute__(self, '_mask')
        _data = np.ndarray.view(self, _localdict['_baseclass'])
        # We want a field
        if isinstance(indx, str):
            # Make sure _sharedmask is True to propagate back to _fieldmask
            # Don't use _set_mask, there are some copies being made that
            # break propagation Don't force the mask to nomask, that wreaks
            # easy masking
            obj = _data[indx].view(ma.MaskedArray)
            obj._mask = _mask[indx]
            obj._sharedmask = True
            fval = _localdict['_fill_value']
            if fval is not None:
                obj._fill_value = fval[indx]
            # Force to masked if the mask is True
            if not obj.ndim and obj._mask:
                return ma.masked
            return obj
        # We want some elements.
        # First, the data.
        obj = np.asarray(_data[indx]).view(mrecarray)
        obj._mask = np.asarray(_mask[indx]).view(np.recarray)
        return obj

    def __setitem__(self, indx, value):
        """
        Sets the given record to value.

        """
        ma.MaskedArray.__setitem__(self, indx, value)
        if isinstance(indx, str):
            self._mask[indx] = ma.getmaskarray(value)

    def __str__(self):
        """
        Calculates the string representation.

        """
        if self.size > 1:
            mstr = [f"({','.join([str(i) for i in s])})"
                    for s in zip(*[getattr(self, f) for f in self.dtype.names])]
            return f"[{', '.join(mstr)}]"
        else:
            mstr = [f"{','.join([str(i) for i in s])}"
                    for s in zip([getattr(self, f) for f in self.dtype.names])]
            return f"({', '.join(mstr)})"

    def __repr__(self):
        """
        Calculates the repr representation.

        """
        _names = self.dtype.names
        fmt = f"%{max(len(n) for n in _names) + 4}s : %s"
        reprstr = [fmt % (f, getattr(self, f)) for f in self.dtype.names]
        reprstr.insert(0, 'masked_records(')
        reprstr.extend([fmt % ('    fill_value', self.fill_value),
                        '              )'])
        return str("\n".join(reprstr))

    def view(self, dtype=None, type=None):
        """
        Returns a view of the mrecarray.

        """
        # OK, basic copy-paste from MaskedArray.view.
        if dtype is None:
            if type is None:
                output = np.ndarray.view(self)
            else:
                output = np.ndarray.view(self, type)
        # Here again.
        elif type is None:
            try:
                if issubclass(dtype, np.ndarray):
                    output = np.ndarray.view(self, dtype)
                else:
                    output = np.ndarray.view(self, dtype)
            # OK, there's the change
            except TypeError:
                dtype = np.dtype(dtype)
                # we need to revert to MaskedArray, but keeping the possibility
                # of subclasses (eg, TimeSeriesRecords), so we'll force a type
                # set to the first parent
                if dtype.fields is None:
                    basetype = self.__class__.__bases__[0]
                    output = self.__array__().view(dtype, basetype)
                    output._update_from(self)
                else:
                    output = np.ndarray.view(self, dtype)
                output._fill_value = None
        else:
            output = np.ndarray.view(self, dtype, type)
        # Update the mask, just like in MaskedArray.view
        if (getattr(output, '_mask', ma.nomask) is not ma.nomask):
            mdtype = ma.make_mask_descr(output.dtype)
            output._mask = self._mask.view(mdtype, np.ndarray)
            output._mask.shape = output.shape
        return output

    def harden_mask(self):
        """
        Forces the mask to hard.

        """
        self._hardmask = True

    def soften_mask(self):
        """
        Forces the mask to soft

        """
        self._hardmask = False

    def copy(self):
        """
        Returns a copy of the masked record.

        """
        copied = self._data.copy().view(type(self))
        copied._mask = self._mask.copy()
        return copied

    def tolist(self, fill_value=None):
        """
        Return the data portion of the array as a list.

        Data items are converted to the nearest compatible Python type.
        Masked values are converted to fill_value. If fill_value is None,
        the corresponding entries in the output list will be ``None``.

        """
        if fill_value is not None:
            return self.filled(fill_value).tolist()
        result = np.array(self.filled().tolist(), dtype=object)
        mask = np.array(self._mask.tolist())
        result[mask] = None
        return result.tolist()

    def __getstate__(self):
        """Return the internal state of the masked array.

        This is for pickling.

        """
        state = (1,
                 self.shape,
                 self.dtype,
                 self.flags.fnc,
                 self._data.tobytes(),
                 self._mask.tobytes(),
                 self._fill_value,
                 )
        return state

    def __setstate__(self, state):
        """
        Restore the internal state of the masked array.

        This is for pickling.  ``state`` is typically the output of the
        ``__getstate__`` output, and is a 5-tuple:

        - class name
        - a tuple giving the shape of the data
        - a typecode for the data
        - a binary string for the data
        - a binary string for the mask.

        """
        (ver, shp, typ, isf, raw, msk, flv) = state
        np.ndarray.__setstate__(self, (shp, typ, isf, raw))
        mdtype = np.dtype([(k, np.bool) for (k, _) in self.dtype.descr])
        self.__dict__['_mask'].__setstate__((shp, mdtype, isf, msk))
        self.fill_value = flv

    def __reduce__(self):
        """
        Return a 3-tuple for pickling a MaskedArray.

        """
        return (_mrreconstruct,
                (self.__class__, self._baseclass, (0,), 'b',),
                self.__getstate__())


def _mrreconstruct(subtype, baseclass, baseshape, basetype,):
    """
    Build a new MaskedArray from the information stored in a pickle.

    """
    _data = np.ndarray.__new__(baseclass, baseshape, basetype).view(subtype)
    _mask = np.ndarray.__new__(np.ndarray, baseshape, 'b1')
    return subtype.__new__(subtype, _data, mask=_mask, dtype=basetype,)


mrecarray = MaskedRecords


###############################################################################
#                             Constructors                                    #
###############################################################################


def fromarrays(arraylist, dtype=None, shape=None, formats=None,
               names=None, titles=None, aligned=False, byteorder=None,
               fill_value=None):
    """
    Creates a mrecarray from a (flat) list of masked arrays.

    Parameters
    ----------
    arraylist : sequence
        A list of (masked) arrays. Each element of the sequence is first converted
        to a masked array if needed. If a 2D array is passed as argument, it is
        processed line by line
    dtype : {None, dtype}, optional
        Data type descriptor.
    shape : {None, integer}, optional
        Number of records. If None, shape is defined from the shape of the
        first array in the list.
    formats : {None, sequence}, optional
        Sequence of formats for each individual field. If None, the formats will
        be autodetected by inspecting the fields and selecting the highest dtype
        possible.
    names : {None, sequence}, optional
        Sequence of the names of each field.
    fill_value : {None, sequence}, optional
        Sequence of data to be used as filling values.

    Notes
    -----
    Lists of tuples should be preferred over lists of lists for faster processing.

    """
    datalist = [ma.getdata(x) for x in arraylist]
    masklist = [np.atleast_1d(ma.getmaskarray(x)) for x in arraylist]
    _array = np.rec.fromarrays(datalist,
                               dtype=dtype, shape=shape, formats=formats,
                               names=names, titles=titles, aligned=aligned,
                               byteorder=byteorder).view(mrecarray)
    _array._mask.flat = list(zip(*masklist))
    if fill_value is not None:
        _array.fill_value = fill_value
    return _array


def fromrecords(reclist, dtype=None, shape=None, formats=None, names=None,
                titles=None, aligned=False, byteorder=None,
                fill_value=None, mask=ma.nomask):
    """
    Creates a MaskedRecords from a list of records.

    Parameters
    ----------
    reclist : sequence
        A list of records. Each element of the sequence is first converted
        to a masked array if needed. If a 2D array is passed as argument, it is
        processed line by line
    dtype : {None, dtype}, optional
        Data type descriptor.
    shape : {None,int}, optional
        Number of records. If None, ``shape`` is defined from the shape of the
        first array in the list.
    formats : {None, sequence}, optional
        Sequence of formats for each individual field. If None, the formats will
        be autodetected by inspecting the fields and selecting the highest dtype
        possible.
    names : {None, sequence}, optional
        Sequence of the names of each field.
    fill_value : {None, sequence}, optional
        Sequence of data to be used as filling values.
    mask : {nomask, sequence}, optional.
        External mask to apply on the data.

    Notes
    -----
    Lists of tuples should be preferred over lists of lists for faster processing.

    """
    # Grab the initial _fieldmask, if needed:
    _mask = getattr(reclist, '_mask', None)
    # Get the list of records.
    if isinstance(reclist, np.ndarray):
        # Make sure we don't have some hidden mask
        if isinstance(reclist, ma.MaskedArray):
            reclist = reclist.filled().view(np.ndarray)
        # Grab the initial dtype, just in case
        if dtype is None:
            dtype = reclist.dtype
        reclist = reclist.tolist()
    mrec = np.rec.fromrecords(reclist, dtype=dtype, shape=shape, formats=formats,
                          names=names, titles=titles,
                          aligned=aligned, byteorder=byteorder).view(mrecarray)
    # Set the fill_value if needed
    if fill_value is not None:
        mrec.fill_value = fill_value
    # Now, let's deal w/ the mask
    if mask is not ma.nomask:
        mask = np.asarray(mask)
        maskrecordlength = len(mask.dtype)
        if maskrecordlength:
            mrec._mask.flat = mask
        elif mask.ndim == 2:
            mrec._mask.flat = [tuple(m) for m in mask]
        else:
            mrec.__setmask__(mask)
    if _mask is not None:
        mrec._mask[:] = _mask
    return mrec


def _guessvartypes(arr):
    """
    Tries to guess the dtypes of the str_ ndarray `arr`.

    Guesses by testing element-wise conversion. Returns a list of dtypes.
    The array is first converted to ndarray. If the array is 2D, the test
    is performed on the first line. An exception is raised if the file is
    3D or more.

    """
    vartypes = []
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[0]
    elif arr.ndim > 2:
        raise ValueError("The array should be 2D at most!")
    # Start the conversion loop.
    for f in arr:
        try:
            int(f)
        except (ValueError, TypeError):
            try:
                float(f)
            except (ValueError, TypeError):
                try:
                    complex(f)
                except (ValueError, TypeError):
                    vartypes.append(arr.dtype)
                else:
                    vartypes.append(np.dtype(complex))
            else:
                vartypes.append(np.dtype(float))
        else:
            vartypes.append(np.dtype(int))
    return vartypes


def openfile(fname):
    """
    Opens the file handle of file `fname`.

    """
    # A file handle
    if hasattr(fname, 'readline'):
        return fname
    # Try to open the file and guess its type
    try:
        f = open(fname)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"No such file: '{fname}'") from e
    if f.readline()[:2] != "\\x":
        f.seek(0, 0)
        return f
    f.close()
    raise NotImplementedError("Wow, binary file")


def fromtextfile(fname, delimiter=None, commentchar='#', missingchar='',
                 varnames=None, vartypes=None,
                 *, delimitor=np._NoValue):  # backwards compatibility
    """
    Creates a mrecarray from data stored in the file `filename`.

    Parameters
    ----------
    fname : {file name/handle}
        Handle of an opened file.
    delimiter : {None, string}, optional
        Alphanumeric character used to separate columns in the file.
        If None, any (group of) white spacestring(s) will be used.
    commentchar : {'#', string}, optional
        Alphanumeric character used to mark the start of a comment.
    missingchar : {'', string}, optional
        String indicating missing data, and used to create the masks.
    varnames : {None, sequence}, optional
        Sequence of the variable names. If None, a list will be created from
        the first non empty line of the file.
    vartypes : {None, sequence}, optional
        Sequence of the variables dtypes. If None, it will be estimated from
        the first non-commented line.


    Ultra simple: the varnames are in the header, one line"""
    if delimitor is not np._NoValue:
        if delimiter is not None:
            raise TypeError("fromtextfile() got multiple values for argument "
                            "'delimiter'")
        # NumPy 1.22.0, 2021-09-23
        warnings.warn("The 'delimitor' keyword argument of "
                      "numpy.ma.mrecords.fromtextfile() is deprecated "
                      "since NumPy 1.22.0, use 'delimiter' instead.",
                      DeprecationWarning, stacklevel=2)
        delimiter = delimitor

    # Try to open the file.
    ftext = openfile(fname)

    # Get the first non-empty line as the varnames
    while True:
        line = ftext.readline()
        firstline = line[:line.find(commentchar)].strip()
        _varnames = firstline.split(delimiter)
        if len(_varnames) > 1:
            break
    if varnames is None:
        varnames = _varnames

    # Get the data.
    _variables = ma.masked_array([line.strip().split(delimiter) for line in ftext
                                 if line[0] != commentchar and len(line) > 1])
    (_, nfields) = _variables.shape
    ftext.close()

    # Try to guess the dtype.
    if vartypes is None:
        vartypes = _guessvartypes(_variables[0])
    else:
        vartypes = [np.dtype(v) for v in vartypes]
        if len(vartypes) != nfields:
            msg = f"Attempting to {len(vartypes)} dtypes for {nfields} fields!"
            msg += " Reverting to default."
            warnings.warn(msg, stacklevel=2)
            vartypes = _guessvartypes(_variables[0])

    # Construct the descriptor.
    mdescr = list(zip(varnames, vartypes))
    mfillv = [ma.default_fill_value(f) for f in vartypes]

    # Get the data and the mask.
    # We just need a list of masked_arrays. It's easier to create it like that:
    _mask = (_variables.T == missingchar)
    _datalist = [ma.masked_array(a, mask=m, dtype=t, fill_value=f)
                 for (a, m, t, f) in zip(_variables.T, _mask, vartypes, mfillv)]

    return fromarrays(_datalist, dtype=mdescr)


def addfield(mrecord, newfield, newfieldname=None):
    """Adds a new field to the masked record array

    Uses `newfield` as data and `newfieldname` as name. If `newfieldname`
    is None, the new field name is set to 'fi', where `i` is the number of
    existing fields.

    """
    _data = mrecord._data
    _mask = mrecord._mask
    if newfieldname is None or newfieldname in reserved_fields:
        newfieldname = f'f{len(_data.dtype)}'
    newfield = ma.array(newfield)
    # Get the new data.
    # Create a new empty recarray
    newdtype = np.dtype(_data.dtype.descr + [(newfieldname, newfield.dtype)])
    newdata = np.recarray(_data.shape, newdtype)
    # Add the existing field
    [newdata.setfield(_data.getfield(*f), *f)
     for f in _data.dtype.fields.values()]
    # Add the new field
    newdata.setfield(newfield._data, *newdata.dtype.fields[newfieldname])
    newdata = newdata.view(MaskedRecords)
    # Get the new mask
    # Create a new empty recarray
    newmdtype = np.dtype([(n, np.bool) for n in newdtype.names])
    newmask = np.recarray(_data.shape, newmdtype)
    # Add the old masks
    [newmask.setfield(_mask.getfield(*f), *f)
     for f in _mask.dtype.fields.values()]
    # Add the mask of the new field
    newmask.setfield(ma.getmaskarray(newfield),
                     *newmask.dtype.fields[newfieldname])
    newdata._mask = newmask
    return newdata

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\utils\misc.py ===
import errno
import getpass
import hashlib
import logging
import os
import posixpath
import shutil
import stat
import sys
import sysconfig
import urllib.parse
from dataclasses import dataclass
from functools import partial
from io import StringIO
from itertools import filterfalse, tee, zip_longest
from pathlib import Path
from types import FunctionType, TracebackType
from typing import (
    Any,
    BinaryIO,
    Callable,
    Generator,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from pip._vendor.packaging.requirements import Requirement
from pip._vendor.pyproject_hooks import BuildBackendHookCaller

from pip import __version__
from pip._internal.exceptions import CommandError, ExternallyManagedEnvironment
from pip._internal.locations import get_major_minor_version
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.retry import retry
from pip._internal.utils.virtualenv import running_under_virtualenv

__all__ = [
    "rmtree",
    "display_path",
    "backup_dir",
    "ask",
    "splitext",
    "format_size",
    "is_installable_dir",
    "normalize_path",
    "renames",
    "get_prog",
    "ensure_dir",
    "remove_auth_from_url",
    "check_externally_managed",
    "ConfiguredBuildBackendHookCaller",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")
ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]
VersionInfo = Tuple[int, int, int]
NetlocTuple = Tuple[str, Tuple[Optional[str], Optional[str]]]
OnExc = Callable[[FunctionType, Path, BaseException], Any]
OnErr = Callable[[FunctionType, Path, ExcInfo], Any]

FILE_CHUNK_SIZE = 1024 * 1024


def get_pip_version() -> str:
    pip_pkg_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    pip_pkg_dir = os.path.abspath(pip_pkg_dir)

    return f"pip {__version__} from {pip_pkg_dir} (python {get_major_minor_version()})"


def normalize_version_info(py_version_info: Tuple[int, ...]) -> Tuple[int, int, int]:
    """
    Convert a tuple of ints representing a Python version to one of length
    three.

    :param py_version_info: a tuple of ints representing a Python version,
        or None to specify no version. The tuple can have any length.

    :return: a tuple of length three if `py_version_info` is non-None.
        Otherwise, return `py_version_info` unchanged (i.e. None).
    """
    if len(py_version_info) < 3:
        py_version_info += (3 - len(py_version_info)) * (0,)
    elif len(py_version_info) > 3:
        py_version_info = py_version_info[:3]

    return cast("VersionInfo", py_version_info)


def ensure_dir(path: str) -> None:
    """os.path.makedirs without EEXIST."""
    try:
        os.makedirs(path)
    except OSError as e:
        # Windows can raise spurious ENOTEMPTY errors. See #6426.
        if e.errno != errno.EEXIST and e.errno != errno.ENOTEMPTY:
            raise


def get_prog() -> str:
    try:
        prog = os.path.basename(sys.argv[0])
        if prog in ("__main__.py", "-c"):
            return f"{sys.executable} -m pip"
        else:
            return prog
    except (AttributeError, TypeError, IndexError):
        pass
    return "pip"


# Retry every half second for up to 3 seconds
@retry(stop_after_delay=3, wait=0.5)
def rmtree(
    dir: str, ignore_errors: bool = False, onexc: Optional[OnExc] = None
) -> None:
    if ignore_errors:
        onexc = _onerror_ignore
    if onexc is None:
        onexc = _onerror_reraise
    handler: OnErr = partial(rmtree_errorhandler, onexc=onexc)
    if sys.version_info >= (3, 12):
        # See https://docs.python.org/3.12/whatsnew/3.12.html#shutil.
        shutil.rmtree(dir, onexc=handler)  # type: ignore
    else:
        shutil.rmtree(dir, onerror=handler)  # type: ignore


def _onerror_ignore(*_args: Any) -> None:
    pass


def _onerror_reraise(*_args: Any) -> None:
    raise  # noqa: PLE0704 - Bare exception used to reraise existing exception


def rmtree_errorhandler(
    func: FunctionType,
    path: Path,
    exc_info: Union[ExcInfo, BaseException],
    *,
    onexc: OnExc = _onerror_reraise,
) -> None:
    """
    `rmtree` error handler to 'force' a file remove (i.e. like `rm -f`).

    * If a file is readonly then it's write flag is set and operation is
      retried.

    * `onerror` is the original callback from `rmtree(... onerror=onerror)`
      that is chained at the end if the "rm -f" still fails.
    """
    try:
        st_mode = os.stat(path).st_mode
    except OSError:
        # it's equivalent to os.path.exists
        return

    if not st_mode & stat.S_IWRITE:
        # convert to read/write
        try:
            os.chmod(path, st_mode | stat.S_IWRITE)
        except OSError:
            pass
        else:
            # use the original function to repeat the operation
            try:
                func(path)
                return
            except OSError:
                pass

    if not isinstance(exc_info, BaseException):
        _, exc_info, _ = exc_info
    onexc(func, path, exc_info)


def display_path(path: str) -> str:
    """Gives the display value for a given path, making it relative to cwd
    if possible."""
    path = os.path.normcase(os.path.abspath(path))
    if path.startswith(os.getcwd() + os.path.sep):
        path = "." + path[len(os.getcwd()) :]
    return path


def backup_dir(dir: str, ext: str = ".bak") -> str:
    """Figure out the name of a directory to back up the given dir to
    (adding .bak, .bak2, etc)"""
    n = 1
    extension = ext
    while os.path.exists(dir + extension):
        n += 1
        extension = ext + str(n)
    return dir + extension


def ask_path_exists(message: str, options: Iterable[str]) -> str:
    for action in os.environ.get("PIP_EXISTS_ACTION", "").split():
        if action in options:
            return action
    return ask(message, options)


def _check_no_input(message: str) -> None:
    """Raise an error if no input is allowed."""
    if os.environ.get("PIP_NO_INPUT"):
        raise Exception(
            f"No input was expected ($PIP_NO_INPUT set); question: {message}"
        )


def ask(message: str, options: Iterable[str]) -> str:
    """Ask the message interactively, with the given possible responses"""
    while 1:
        _check_no_input(message)
        response = input(message)
        response = response.strip().lower()
        if response not in options:
            print(
                "Your response ({!r}) was not one of the expected responses: "
                "{}".format(response, ", ".join(options))
            )
        else:
            return response


def ask_input(message: str) -> str:
    """Ask for input interactively."""
    _check_no_input(message)
    return input(message)


def ask_password(message: str) -> str:
    """Ask for a password interactively."""
    _check_no_input(message)
    return getpass.getpass(message)


def strtobool(val: str) -> int:
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif val in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError(f"invalid truth value {val!r}")


def format_size(bytes: float) -> str:
    if bytes > 1000 * 1000:
        return f"{bytes / 1000.0 / 1000:.1f} MB"
    elif bytes > 10 * 1000:
        return f"{int(bytes / 1000)} kB"
    elif bytes > 1000:
        return f"{bytes / 1000.0:.1f} kB"
    else:
        return f"{int(bytes)} bytes"


def tabulate(rows: Iterable[Iterable[Any]]) -> Tuple[List[str], List[int]]:
    """Return a list of formatted rows and a list of column sizes.

    For example::

    >>> tabulate([['foobar', 2000], [0xdeadbeef]])
    (['foobar     2000', '3735928559'], [10, 4])
    """
    rows = [tuple(map(str, row)) for row in rows]
    sizes = [max(map(len, col)) for col in zip_longest(*rows, fillvalue="")]
    table = [" ".join(map(str.ljust, row, sizes)).rstrip() for row in rows]
    return table, sizes


def is_installable_dir(path: str) -> bool:
    """Is path is a directory containing pyproject.toml or setup.py?

    If pyproject.toml exists, this is a PEP 517 project. Otherwise we look for
    a legacy setuptools layout by identifying setup.py. We don't check for the
    setup.cfg because using it without setup.py is only available for PEP 517
    projects, which are already covered by the pyproject.toml check.
    """
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "pyproject.toml")):
        return True
    if os.path.isfile(os.path.join(path, "setup.py")):
        return True
    return False


def read_chunks(
    file: BinaryIO, size: int = FILE_CHUNK_SIZE
) -> Generator[bytes, None, None]:
    """Yield pieces of data from a file-like object until EOF."""
    while True:
        chunk = file.read(size)
        if not chunk:
            break
        yield chunk


def normalize_path(path: str, resolve_symlinks: bool = True) -> str:
    """
    Convert a path to its canonical, case-normalized, absolute version.

    """
    path = os.path.expanduser(path)
    if resolve_symlinks:
        path = os.path.realpath(path)
    else:
        path = os.path.abspath(path)
    return os.path.normcase(path)


def splitext(path: str) -> Tuple[str, str]:
    """Like os.path.splitext, but take off .tar too"""
    base, ext = posixpath.splitext(path)
    if base.lower().endswith(".tar"):
        ext = base[-4:] + ext
        base = base[:-4]
    return base, ext


def renames(old: str, new: str) -> None:
    """Like os.renames(), but handles renaming across devices."""
    # Implementation borrowed from os.renames().
    head, tail = os.path.split(new)
    if head and tail and not os.path.exists(head):
        os.makedirs(head)

    shutil.move(old, new)

    head, tail = os.path.split(old)
    if head and tail:
        try:
            os.removedirs(head)
        except OSError:
            pass


def is_local(path: str) -> bool:
    """
    Return True if path is within sys.prefix, if we're running in a virtualenv.

    If we're not in a virtualenv, all paths are considered "local."

    Caution: this function assumes the head of path has been normalized
    with normalize_path.
    """
    if not running_under_virtualenv():
        return True
    return path.startswith(normalize_path(sys.prefix))


def write_output(msg: Any, *args: Any) -> None:
    logger.info(msg, *args)


class StreamWrapper(StringIO):
    orig_stream: TextIO

    @classmethod
    def from_stream(cls, orig_stream: TextIO) -> "StreamWrapper":
        ret = cls()
        ret.orig_stream = orig_stream
        return ret

    # compileall.compile_dir() needs stdout.encoding to print to stdout
    # type ignore is because TextIOBase.encoding is writeable
    @property
    def encoding(self) -> str:  # type: ignore
        return self.orig_stream.encoding


# Simulates an enum
def enum(*sequential: Any, **named: Any) -> Type[Any]:
    enums = dict(zip(sequential, range(len(sequential))), **named)
    reverse = {value: key for key, value in enums.items()}
    enums["reverse_mapping"] = reverse
    return type("Enum", (), enums)


def build_netloc(host: str, port: Optional[int]) -> str:
    """
    Build a netloc from a host-port pair
    """
    if port is None:
        return host
    if ":" in host:
        # Only wrap host with square brackets when it is IPv6
        host = f"[{host}]"
    return f"{host}:{port}"


def build_url_from_netloc(netloc: str, scheme: str = "https") -> str:
    """
    Build a full URL from a netloc.
    """
    if netloc.count(":") >= 2 and "@" not in netloc and "[" not in netloc:
        # It must be a bare IPv6 address, so wrap it with brackets.
        netloc = f"[{netloc}]"
    return f"{scheme}://{netloc}"


def parse_netloc(netloc: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Return the host-port pair from a netloc.
    """
    url = build_url_from_netloc(netloc)
    parsed = urllib.parse.urlparse(url)
    return parsed.hostname, parsed.port


def split_auth_from_netloc(netloc: str) -> NetlocTuple:
    """
    Parse out and remove the auth information from a netloc.

    Returns: (netloc, (username, password)).
    """
    if "@" not in netloc:
        return netloc, (None, None)

    # Split from the right because that's how urllib.parse.urlsplit()
    # behaves if more than one @ is present (which can be checked using
    # the password attribute of urlsplit()'s return value).
    auth, netloc = netloc.rsplit("@", 1)
    pw: Optional[str] = None
    if ":" in auth:
        # Split from the left because that's how urllib.parse.urlsplit()
        # behaves if more than one : is present (which again can be checked
        # using the password attribute of the return value)
        user, pw = auth.split(":", 1)
    else:
        user, pw = auth, None

    user = urllib.parse.unquote(user)
    if pw is not None:
        pw = urllib.parse.unquote(pw)

    return netloc, (user, pw)


def redact_netloc(netloc: str) -> str:
    """
    Replace the sensitive data in a netloc with "****", if it exists.

    For example:
        - "user:pass@example.com" returns "user:****@example.com"
        - "accesstoken@example.com" returns "****@example.com"
    """
    netloc, (user, password) = split_auth_from_netloc(netloc)
    if user is None:
        return netloc
    if password is None:
        user = "****"
        password = ""
    else:
        user = urllib.parse.quote(user)
        password = ":****"
    return f"{user}{password}@{netloc}"


def _transform_url(
    url: str, transform_netloc: Callable[[str], Tuple[Any, ...]]
) -> Tuple[str, NetlocTuple]:
    """Transform and replace netloc in a url.

    transform_netloc is a function taking the netloc and returning a
    tuple. The first element of this tuple is the new netloc. The
    entire tuple is returned.

    Returns a tuple containing the transformed url as item 0 and the
    original tuple returned by transform_netloc as item 1.
    """
    purl = urllib.parse.urlsplit(url)
    netloc_tuple = transform_netloc(purl.netloc)
    # stripped url
    url_pieces = (purl.scheme, netloc_tuple[0], purl.path, purl.query, purl.fragment)
    surl = urllib.parse.urlunsplit(url_pieces)
    return surl, cast("NetlocTuple", netloc_tuple)


def _get_netloc(netloc: str) -> NetlocTuple:
    return split_auth_from_netloc(netloc)


def _redact_netloc(netloc: str) -> Tuple[str]:
    return (redact_netloc(netloc),)


def split_auth_netloc_from_url(
    url: str,
) -> Tuple[str, str, Tuple[Optional[str], Optional[str]]]:
    """
    Parse a url into separate netloc, auth, and url with no auth.

    Returns: (url_without_auth, netloc, (username, password))
    """
    url_without_auth, (netloc, auth) = _transform_url(url, _get_netloc)
    return url_without_auth, netloc, auth


def remove_auth_from_url(url: str) -> str:
    """Return a copy of url with 'username:password@' removed."""
    # username/pass params are passed to subversion through flags
    # and are not recognized in the url.
    return _transform_url(url, _get_netloc)[0]


def redact_auth_from_url(url: str) -> str:
    """Replace the password in a given url with ****."""
    return _transform_url(url, _redact_netloc)[0]


def redact_auth_from_requirement(req: Requirement) -> str:
    """Replace the password in a given requirement url with ****."""
    if not req.url:
        return str(req)
    return str(req).replace(req.url, redact_auth_from_url(req.url))


@dataclass(frozen=True)
class HiddenText:
    secret: str
    redacted: str

    def __repr__(self) -> str:
        return f"<HiddenText {str(self)!r}>"

    def __str__(self) -> str:
        return self.redacted

    # This is useful for testing.
    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return False

        # The string being used for redaction doesn't also have to match,
        # just the raw, original string.
        return self.secret == other.secret


def hide_value(value: str) -> HiddenText:
    return HiddenText(value, redacted="****")


def hide_url(url: str) -> HiddenText:
    redacted = redact_auth_from_url(url)
    return HiddenText(url, redacted=redacted)


def protect_pip_from_modification_on_windows(modifying_pip: bool) -> None:
    """Protection of pip.exe from modification on Windows

    On Windows, any operation modifying pip should be run as:
        python -m pip ...
    """
    pip_names = [
        "pip",
        f"pip{sys.version_info.major}",
        f"pip{sys.version_info.major}.{sys.version_info.minor}",
    ]

    # See https://github.com/pypa/pip/issues/1299 for more discussion
    should_show_use_python_msg = (
        modifying_pip and WINDOWS and os.path.basename(sys.argv[0]) in pip_names
    )

    if should_show_use_python_msg:
        new_command = [sys.executable, "-m", "pip"] + sys.argv[1:]
        raise CommandError(
            "To modify pip, please run the following command:\n{}".format(
                " ".join(new_command)
            )
        )


def check_externally_managed() -> None:
    """Check whether the current environment is externally managed.

    If the ``EXTERNALLY-MANAGED`` config file is found, the current environment
    is considered externally managed, and an ExternallyManagedEnvironment is
    raised.
    """
    if running_under_virtualenv():
        return
    marker = os.path.join(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED")
    if not os.path.isfile(marker):
        return
    raise ExternallyManagedEnvironment.from_config(marker)


def is_console_interactive() -> bool:
    """Is this console interactive?"""
    return sys.stdin is not None and sys.stdin.isatty()


def hash_file(path: str, blocksize: int = 1 << 20) -> Tuple[Any, int]:
    """Return (hash, length) for path using hashlib.sha256()"""

    h = hashlib.sha256()
    length = 0
    with open(path, "rb") as f:
        for block in read_chunks(f, size=blocksize):
            length += len(block)
            h.update(block)
    return h, length


def pairwise(iterable: Iterable[Any]) -> Iterator[Tuple[Any, Any]]:
    """
    Return paired elements.

    For example:
        s -> (s0, s1), (s2, s3), (s4, s5), ...
    """
    iterable = iter(iterable)
    return zip_longest(iterable, iterable)


def partition(
    pred: Callable[[T], bool], iterable: Iterable[T]
) -> Tuple[Iterable[T], Iterable[T]]:
    """
    Use a predicate to partition entries into false entries and true entries,
    like

        partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    """
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)


class ConfiguredBuildBackendHookCaller(BuildBackendHookCaller):
    def __init__(
        self,
        config_holder: Any,
        source_dir: str,
        build_backend: str,
        backend_path: Optional[str] = None,
        runner: Optional[Callable[..., None]] = None,
        python_executable: Optional[str] = None,
    ):
        super().__init__(
            source_dir, build_backend, backend_path, runner, python_executable
        )
        self.config_holder = config_holder

    def build_wheel(
        self,
        wheel_directory: str,
        config_settings: Optional[Mapping[str, Any]] = None,
        metadata_directory: Optional[str] = None,
    ) -> str:
        cs = self.config_holder.config_settings
        return super().build_wheel(
            wheel_directory, config_settings=cs, metadata_directory=metadata_directory
        )

    def build_sdist(
        self,
        sdist_directory: str,
        config_settings: Optional[Mapping[str, Any]] = None,
    ) -> str:
        cs = self.config_holder.config_settings
        return super().build_sdist(sdist_directory, config_settings=cs)

    def build_editable(
        self,
        wheel_directory: str,
        config_settings: Optional[Mapping[str, Any]] = None,
        metadata_directory: Optional[str] = None,
    ) -> str:
        cs = self.config_holder.config_settings
        return super().build_editable(
            wheel_directory, config_settings=cs, metadata_directory=metadata_directory
        )

    def get_requires_for_build_wheel(
        self, config_settings: Optional[Mapping[str, Any]] = None
    ) -> Sequence[str]:
        cs = self.config_holder.config_settings
        return super().get_requires_for_build_wheel(config_settings=cs)

    def get_requires_for_build_sdist(
        self, config_settings: Optional[Mapping[str, Any]] = None
    ) -> Sequence[str]:
        cs = self.config_holder.config_settings
        return super().get_requires_for_build_sdist(config_settings=cs)

    def get_requires_for_build_editable(
        self, config_settings: Optional[Mapping[str, Any]] = None
    ) -> Sequence[str]:
        cs = self.config_holder.config_settings
        return super().get_requires_for_build_editable(config_settings=cs)

    def prepare_metadata_for_build_wheel(
        self,
        metadata_directory: str,
        config_settings: Optional[Mapping[str, Any]] = None,
        _allow_fallback: bool = True,
    ) -> str:
        cs = self.config_holder.config_settings
        return super().prepare_metadata_for_build_wheel(
            metadata_directory=metadata_directory,
            config_settings=cs,
            _allow_fallback=_allow_fallback,
        )

    def prepare_metadata_for_build_editable(
        self,
        metadata_directory: str,
        config_settings: Optional[Mapping[str, Any]] = None,
        _allow_fallback: bool = True,
    ) -> Optional[str]:
        cs = self.config_holder.config_settings
        return super().prepare_metadata_for_build_editable(
            metadata_directory=metadata_directory,
            config_settings=cs,
            _allow_fallback=_allow_fallback,
        )


def warn_if_run_as_root() -> None:
    """Output a warning for sudo users on Unix.

    In a virtual environment, sudo pip still writes to virtualenv.
    On Windows, users may run pip as Administrator without issues.
    This warning only applies to Unix root users outside of virtualenv.
    """
    if running_under_virtualenv():
        return
    if not hasattr(os, "getuid"):
        return
    # On Windows, there are no "system managed" Python packages. Installing as
    # Administrator via pip is the correct way of updating system environments.
    #
    # We choose sys.platform over utils.compat.WINDOWS here to enable Mypy platform
    # checks: https://mypy.readthedocs.io/en/stable/common_issues.html
    if sys.platform == "win32" or sys.platform == "cygwin":
        return

    if os.getuid() != 0:
        return

    logger.warning(
        "Running pip as the 'root' user can result in broken permissions and "
        "conflicting behaviour with the system package manager, possibly "
        "rendering your system unusable. "
        "It is recommended to use a virtual environment instead: "
        "https://pip.pypa.io/warnings/venv. "
        "Use the --root-user-action option if you know what you are doing and "
        "want to suppress this warning."
    )