# Project Analysis Report
- **Project**: atelier-kyo-manager
- **Profile**: AI Analysis Pack (for Gemini)
- **Export Mode**: Gemini Optimized
- **Timestamp**: 2025-08-24T11:55:11.935116

## 📊 Statistics
- **Total Files**: 691
- **Total Lines**: 303,357

| Extension | Files | Lines of Code |
|---|---|---|
| `.py` | 658 | 300,080 |
| `.md` | 17 | 739 |
| `.html` | 6 | 0 |
| `.txt` | 5 | 41 |
| `.toml` | 4 | 2,497 |
| `.css` | 1 | 0 |


## 🌳 Directory Structure
```
📁 **atelier-kyo-manager/**
├── .venv
│   └── Lib
│       └── site-packages
│           ├── flask
│           │   └── sansio
│           │       └── README.md
│           ├── gradio
│           │   ├── _frontend_code
│           │   │   ├── client
│           │   │   │   └── README.md
│           │   │   ├── lite
│           │   │   │   └── examples
│           │   │   │       └── transformers_basic
│           │   │   │           └── requirements.txt
│           │   │   └── preview
│           │   │       └── test
│           │   │           └── test
│           │   │               └── pyproject.toml
│           │   ├── cli
│           │   │   └── commands
│           │   │       └── components
│           │   │           └── files
│           │   │               └── README.md
│           │   └── icons
│           │       └── README.md
│           ├── onnx
│           │   └── backend
│           │       └── test
│           │           └── data
│           │               └── light
│           │                   └── README.md
│           ├── pandas
│           │   └── pyproject.toml
│           ├── playwright
│           │   └── driver
│           │       ├── README.md
│           │       └── package
│           │           └── README.md
│           ├── torchgen
│           │   └── packaged
│           │       └── autograd
│           │           └── README.md
│           └── wtforms
│               └── locale
│                   └── README.md
├── .venv_backup
│   └── Lib
│       └── site-packages
│           ├── flask
│           │   └── sansio
│           │       └── README.md
│           ├── google
│           │   └── protobuf
│           │       ├── compiler
│           │       │   └── __init__.py
│           │       ├── pyext
│           │       │   └── __init__.py
│           │       ├── testdata
│           │       │   └── __init__.py
│           │       └── util
│           │           └── __init__.py
│           ├── greenlet
│           │   └── platform
│           │       └── __init__.py
│           ├── mpmath
│           │   └── tests
│           │       └── __init__.py
│           ├── numpy
│           │   ├── _pyinstaller
│           │   │   └── __init__.py
│           │   ├── fft
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── lib
│           │   │   ├── __init__.py
│           │   │   ├── _array_utils_impl.py
│           │   │   ├── _arraypad_impl.py
│           │   │   ├── _arraysetops_impl.py
│           │   │   ├── _arrayterator_impl.py
│           │   │   ├── _datasource.py
│           │   │   ├── _format_impl.py
│           │   │   ├── _function_base_impl.py
│           │   │   ├── _histograms_impl.py
│           │   │   ├── _index_tricks_impl.py
│           │   │   ├── _iotools.py
│           │   │   ├── _nanfunctions_impl.py
│           │   │   ├── _npyio_impl.py
│           │   │   ├── _polynomial_impl.py
│           │   │   ├── _scimath_impl.py
│           │   │   ├── _shape_base_impl.py
│           │   │   ├── _stride_tricks_impl.py
│           │   │   ├── _twodim_base_impl.py
│           │   │   ├── _type_check_impl.py
│           │   │   ├── _ufunclike_impl.py
│           │   │   ├── _user_array_impl.py
│           │   │   ├── _utils_impl.py
│           │   │   ├── _version.py
│           │   │   ├── array_utils.py
│           │   │   ├── introspect.py
│           │   │   ├── mixins.py
│           │   │   ├── recfunctions.py
│           │   │   └── tests
│           │   │       ├── __init__.py
│           │   │       ├── test__datasource.py
│           │   │       ├── test__iotools.py
│           │   │       ├── test__version.py
│           │   │       ├── test_array_utils.py
│           │   │       ├── test_arraypad.py
│           │   │       ├── test_arraysetops.py
│           │   │       ├── test_format.py
│           │   │       ├── test_function_base.py
│           │   │       ├── test_histograms.py
│           │   │       ├── test_index_tricks.py
│           │   │       ├── test_io.py
│           │   │       ├── test_loadtxt.py
│           │   │       ├── test_mixins.py
│           │   │       ├── test_nanfunctions.py
│           │   │       ├── test_packbits.py
│           │   │       ├── test_polynomial.py
│           │   │       ├── test_recfunctions.py
│           │   │       ├── test_regression.py
│           │   │       ├── test_shape_base.py
│           │   │       ├── test_stride_tricks.py
│           │   │       ├── test_twodim_base.py
│           │   │       ├── test_type_check.py
│           │   │       ├── test_ufunclike.py
│           │   │       └── test_utils.py
│           │   ├── linalg
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── ma
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── matrixlib
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── polynomial
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── random
│           │   │   └── tests
│           │   │       ├── __init__.py
│           │   │       └── data
│           │   │           └── __init__.py
│           │   ├── testing
│           │   │   ├── _private
│           │   │   │   └── __init__.py
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── tests
│           │   │   └── __init__.py
│           │   └── typing
│           │       └── tests
│           │           └── __init__.py
│           ├── onnxruntime
│           │   ├── quantization
│           │   │   └── CalTableFlatBuffers
│           │   │       └── __init__.py
│           │   └── tools
│           │       ├── mobile_helpers
│           │       │   └── __init__.py
│           │       ├── ort_format_model
│           │       │   └── ort_flatbuffers_py
│           │       │       └── __init__.py
│           │       └── qdq_helpers
│           │           └── __init__.py
│           ├── pandas
│           │   ├── _libs
│           │   │   └── window
│           │   │       └── __init__.py
│           │   ├── core
│           │   │   ├── __init__.py
│           │   │   ├── _numba
│           │   │   │   └── __init__.py
│           │   │   ├── computation
│           │   │   │   └── __init__.py
│           │   │   ├── dtypes
│           │   │   │   └── __init__.py
│           │   │   ├── indexes
│           │   │   │   └── __init__.py
│           │   │   ├── interchange
│           │   │   │   └── __init__.py
│           │   │   ├── methods
│           │   │   │   └── __init__.py
│           │   │   ├── reshape
│           │   │   │   └── __init__.py
│           │   │   ├── sparse
│           │   │   │   └── __init__.py
│           │   │   ├── tools
│           │   │   │   └── __init__.py
│           │   │   └── util
│           │   │       └── __init__.py
│           │   ├── pyproject.toml
│           │   └── tests
│           │       ├── __init__.py
│           │       ├── api
│           │       │   └── __init__.py
│           │       ├── apply
│           │       │   └── __init__.py
│           │       ├── arithmetic
│           │       │   └── __init__.py
│           │       ├── arrays
│           │       │   ├── __init__.py
│           │       │   ├── boolean
│           │       │   │   └── __init__.py
│           │       │   ├── categorical
│           │       │   │   └── __init__.py
│           │       │   ├── datetimes
│           │       │   │   └── __init__.py
│           │       │   ├── floating
│           │       │   │   └── __init__.py
│           │       │   ├── integer
│           │       │   │   └── __init__.py
│           │       │   ├── interval
│           │       │   │   └── __init__.py
│           │       │   ├── masked
│           │       │   │   └── __init__.py
│           │       │   ├── numpy_
│           │       │   │   └── __init__.py
│           │       │   ├── period
│           │       │   │   └── __init__.py
│           │       │   ├── sparse
│           │       │   │   └── __init__.py
│           │       │   ├── string_
│           │       │   │   └── __init__.py
│           │       │   └── timedeltas
│           │       │       └── __init__.py
│           │       ├── base
│           │       │   └── __init__.py
│           │       ├── computation
│           │       │   └── __init__.py
│           │       ├── config
│           │       │   └── __init__.py
│           │       ├── construction
│           │       │   └── __init__.py
│           │       ├── copy_view
│           │       │   ├── __init__.py
│           │       │   └── index
│           │       │       └── __init__.py
│           │       ├── dtypes
│           │       │   ├── __init__.py
│           │       │   └── cast
│           │       │       └── __init__.py
│           │       ├── extension
│           │       │   └── __init__.py
│           │       ├── frame
│           │       │   ├── __init__.py
│           │       │   ├── constructors
│           │       │   │   └── __init__.py
│           │       │   └── indexing
│           │       │       └── __init__.py
│           │       ├── generic
│           │       │   └── __init__.py
│           │       ├── groupby
│           │       │   ├── aggregate
│           │       │   │   └── __init__.py
│           │       │   ├── methods
│           │       │   │   └── __init__.py
│           │       │   └── transform
│           │       │       └── __init__.py
│           │       ├── indexes
│           │       │   ├── __init__.py
│           │       │   ├── base_class
│           │       │   │   └── __init__.py
│           │       │   ├── categorical
│           │       │   │   └── __init__.py
│           │       │   ├── datetimelike_
│           │       │   │   └── __init__.py
│           │       │   ├── datetimes
│           │       │   │   ├── __init__.py
│           │       │   │   └── methods
│           │       │   │       └── __init__.py
│           │       │   ├── interval
│           │       │   │   └── __init__.py
│           │       │   ├── multi
│           │       │   │   └── __init__.py
│           │       │   ├── numeric
│           │       │   │   └── __init__.py
│           │       │   ├── object
│           │       │   │   └── __init__.py
│           │       │   ├── period
│           │       │   │   ├── __init__.py
│           │       │   │   └── methods
│           │       │   │       └── __init__.py
│           │       │   ├── ranges
│           │       │   │   └── __init__.py
│           │       │   ├── string
│           │       │   │   └── __init__.py
│           │       │   └── timedeltas
│           │       │       ├── __init__.py
│           │       │       └── methods
│           │       │           └── __init__.py
│           │       ├── indexing
│           │       │   ├── __init__.py
│           │       │   ├── interval
│           │       │   │   └── __init__.py
│           │       │   └── multiindex
│           │       │       └── __init__.py
│           │       ├── interchange
│           │       │   └── __init__.py
│           │       ├── internals
│           │       │   └── __init__.py
│           │       ├── io
│           │       │   ├── __init__.py
│           │       │   ├── excel
│           │       │   │   └── __init__.py
│           │       │   ├── formats
│           │       │   │   ├── __init__.py
│           │       │   │   └── style
│           │       │   │       └── __init__.py
│           │       │   ├── json
│           │       │   │   └── __init__.py
│           │       │   ├── parser
│           │       │   │   ├── __init__.py
│           │       │   │   ├── common
│           │       │   │   │   └── __init__.py
│           │       │   │   ├── dtypes
│           │       │   │   │   └── __init__.py
│           │       │   │   └── usecols
│           │       │   │       └── __init__.py
│           │       │   ├── pytables
│           │       │   │   └── __init__.py
│           │       │   ├── sas
│           │       │   │   └── __init__.py
│           │       │   └── xml
│           │       │       └── __init__.py
│           │       ├── libs
│           │       │   └── __init__.py
│           │       ├── plotting
│           │       │   ├── __init__.py
│           │       │   └── frame
│           │       │       └── __init__.py
│           │       ├── resample
│           │       │   └── __init__.py
│           │       ├── reshape
│           │       │   ├── __init__.py
│           │       │   ├── concat
│           │       │   │   └── __init__.py
│           │       │   └── merge
│           │       │       └── __init__.py
│           │       ├── scalar
│           │       │   ├── __init__.py
│           │       │   ├── interval
│           │       │   │   └── __init__.py
│           │       │   ├── period
│           │       │   │   └── __init__.py
│           │       │   ├── timedelta
│           │       │   │   ├── __init__.py
│           │       │   │   └── methods
│           │       │   │       └── __init__.py
│           │       │   └── timestamp
│           │       │       ├── __init__.py
│           │       │       └── methods
│           │       │           └── __init__.py
│           │       ├── series
│           │       │   ├── __init__.py
│           │       │   ├── accessors
│           │       │   │   └── __init__.py
│           │       │   └── indexing
│           │       │       └── __init__.py
│           │       ├── tools
│           │       │   └── __init__.py
│           │       ├── tseries
│           │       │   ├── __init__.py
│           │       │   ├── frequencies
│           │       │   │   └── __init__.py
│           │       │   ├── holiday
│           │       │   │   └── __init__.py
│           │       │   └── offsets
│           │       │       └── __init__.py
│           │       ├── tslibs
│           │       │   └── __init__.py
│           │       ├── util
│           │       │   └── __init__.py
│           │       └── window
│           │           ├── __init__.py
│           │           └── moments
│           │               └── __init__.py
│           ├── pip
│           │   ├── _internal
│           │   │   ├── operations
│           │   │   │   ├── __init__.py
│           │   │   │   └── build
│           │   │   │       └── __init__.py
│           │   │   ├── resolution
│           │   │   │   ├── __init__.py
│           │   │   │   ├── legacy
│           │   │   │   │   └── __init__.py
│           │   │   │   └── resolvelib
│           │   │   │       └── __init__.py
│           │   │   └── utils
│           │   │       └── __init__.py
│           │   └── _vendor
│           │       └── urllib3
│           │           ├── contrib
│           │           │   ├── __init__.py
│           │           │   └── _securetransport
│           │           │       └── __init__.py
│           │           └── packages
│           │               ├── __init__.py
│           │               └── backports
│           │                   └── __init__.py
│           ├── pyreadline3
│           │   └── lineeditor
│           │       └── __init__.py
│           ├── sniffio
│           │   └── _tests
│           │       └── __init__.py
│           ├── sympy
│           │   ├── algebras
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── assumptions
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── benchmarks
│           │   │   └── __init__.py
│           │   ├── calculus
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── categories
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── codegen
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── combinatorics
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── concrete
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── core
│           │   │   ├── benchmarks
│           │   │   │   └── __init__.py
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── crypto
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── diffgeom
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── discrete
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── external
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── functions
│           │   │   ├── combinatorial
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   ├── elementary
│           │   │   │   ├── benchmarks
│           │   │   │   │   └── __init__.py
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   └── special
│           │   │       ├── benchmarks
│           │   │       │   └── __init__.py
│           │   │       └── tests
│           │   │           └── __init__.py
│           │   ├── geometry
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── holonomic
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── integrals
│           │   │   ├── benchmarks
│           │   │   │   └── __init__.py
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── interactive
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── liealgebras
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── logic
│           │   │   ├── algorithms
│           │   │   │   └── __init__.py
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── matrices
│           │   │   ├── benchmarks
│           │   │   │   └── __init__.py
│           │   │   ├── expressions
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── multipledispatch
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── ntheory
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── parsing
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── physics
│           │   │   ├── biomechanics
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   ├── continuum_mechanics
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   ├── control
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   ├── hep
│           │   │   │   ├── __init__.py
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   ├── mechanics
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   ├── optics
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   ├── quantum
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   ├── tests
│           │   │   │   └── __init__.py
│           │   │   ├── units
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   └── vector
│           │   │       └── tests
│           │   │           └── __init__.py
│           │   ├── plotting
│           │   │   ├── backends
│           │   │   │   └── __init__.py
│           │   │   ├── intervalmath
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   ├── pygletplot
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── polys
│           │   │   ├── agca
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   ├── benchmarks
│           │   │   │   └── __init__.py
│           │   │   ├── domains
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   ├── matrices
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   ├── numberfields
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── printing
│           │   │   ├── pretty
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── sandbox
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── series
│           │   │   ├── benchmarks
│           │   │   │   └── __init__.py
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── sets
│           │   │   ├── handlers
│           │   │   │   └── __init__.py
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── simplify
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── solvers
│           │   │   ├── benchmarks
│           │   │   │   └── __init__.py
│           │   │   ├── diophantine
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   ├── ode
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── stats
│           │   │   ├── sampling
│           │   │   │   ├── __init__.py
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── strategies
│           │   │   ├── branch
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── tensor
│           │   │   ├── array
│           │   │   │   ├── expressions
│           │   │   │   │   └── tests
│           │   │   │   │       └── __init__.py
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── testing
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── unify
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   ├── utilities
│           │   │   ├── _compilation
│           │   │   │   └── tests
│           │   │   │       └── __init__.py
│           │   │   ├── mathml
│           │   │   │   └── data
│           │   │   │       └── __init__.py
│           │   │   └── tests
│           │   │       └── __init__.py
│           │   └── vector
│           │       └── tests
│           │           └── __init__.py
│           ├── trio
│           │   ├── _core
│           │   │   └── _tests
│           │   │       └── __init__.py
│           │   ├── _tests
│           │   │   ├── __init__.py
│           │   │   └── tools
│           │   │       └── __init__.py
│           │   └── _tools
│           │       └── __init__.py
│           ├── tzdata
│           │   └── zoneinfo
│           │       ├── Africa
│           │       │   └── __init__.py
│           │       ├── America
│           │       │   ├── Argentina
│           │       │   │   └── __init__.py
│           │       │   ├── Indiana
│           │       │   │   └── __init__.py
│           │       │   ├── Kentucky
│           │       │   │   └── __init__.py
│           │       │   ├── North_Dakota
│           │       │   │   └── __init__.py
│           │       │   └── __init__.py
│           │       ├── Antarctica
│           │       │   └── __init__.py
│           │       ├── Arctic
│           │       │   └── __init__.py
│           │       ├── Asia
│           │       │   └── __init__.py
│           │       ├── Atlantic
│           │       │   └── __init__.py
│           │       ├── Australia
│           │       │   └── __init__.py
│           │       ├── Brazil
│           │       │   └── __init__.py
│           │       ├── Canada
│           │       │   └── __init__.py
│           │       ├── Chile
│           │       │   └── __init__.py
│           │       ├── Etc
│           │       │   └── __init__.py
│           │       ├── Europe
│           │       │   └── __init__.py
│           │       ├── Indian
│           │       │   └── __init__.py
│           │       ├── Mexico
│           │       │   └── __init__.py
│           │       ├── Pacific
│           │       │   └── __init__.py
│           │       ├── US
│           │       │   └── __init__.py
│           │       └── __init__.py
│           ├── urllib3
│           │   └── contrib
│           │       └── __init__.py
│           ├── webdriver_manager
│           │   ├── core
│           │   │   └── __init__.py
│           │   └── drivers
│           │       └── __init__.py
│           ├── websocket
│           │   └── tests
│           │       └── __init__.py
│           ├── werkzeug
│           │   ├── middleware
│           │   │   └── __init__.py
│           │   └── sansio
│           │       └── __init__.py
│           └── wtforms
│               ├── csrf
│               │   └── __init__.py
│               └── locale
│                   └── README.md
├── README.md
├── app
│   ├── __init__.py
│   ├── extensions.py
│   ├── forms.py
│   ├── models.py
│   ├── models_backup.py
│   ├── routes.py
│   ├── routes_backup.py
│   ├── static
│   │   └── tailwind.css
│   └── utils
│       ├── ai_background_remover.py
│       ├── ai_generate_descriptions.py
│       ├── ai_image_collector.py
│       ├── ai_image_crawler.py
│       ├── ai_llm_controller.py
│       ├── ai_research_orchestrator.py
│       ├── buyma_catalog_manager.py
│       ├── csv_handler.py
│       ├── models
│       │   └── ai_model_builder.py
│       ├── pricing_calculator.py
│       ├── routes_backup_v2.py
│       └── shipping_agent.py
├── config.py
├── debug_catalog
│   ├── debug_catalog_80877.html
│   ├── debug_catalog_80894_20250609_070002.html
│   ├── error_80863_20250606_172521.html
│   ├── error_80894_20250609_070003.html
│   ├── error_main_20250606_172522.html
│   └── error_main_20250609_070003.html
├── export
│   └── atelier-kyo-manager_export_20250811_005523
│       ├── combined_10.py
│       ├── combined_129.py
│       ├── combined_140.py
│       ├── combined_143.py
│       ├── combined_17.py
│       ├── combined_29.py
│       ├── combined_37.py
│       ├── combined_47.py
│       ├── combined_5.py
│       ├── combined_60.py
│       ├── combined_7.py
│       ├── combined_71.py
│       ├── combined_8.py
│       ├── combined_9.py
│       ├── combined_98.py
│       └── source_code
│           └── atelier-kyo-manager
│               ├── .venv_backup
│               │   └── Lib
│               │       └── site-packages
│               │           ├── flask
│               │           │   └── sansio
│               │           │       └── README.md
│               │           ├── google
│               │           │   └── protobuf
│               │           │       ├── compiler
│               │           │       │   └── __init__.py
│               │           │       ├── pyext
│               │           │       │   └── __init__.py
│               │           │       ├── testdata
│               │           │       │   └── __init__.py
│               │           │       └── util
│               │           │           └── __init__.py
│               │           ├── greenlet
│               │           │   └── platform
│               │           │       └── __init__.py
│               │           ├── mpmath
│               │           │   └── tests
│               │           │       └── __init__.py
│               │           ├── numpy
│               │           │   ├── _pyinstaller
│               │           │   │   └── __init__.py
│               │           │   ├── fft
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── lib
│               │           │   │   ├── __init__.py
│               │           │   │   ├── _arraypad_impl.py
│               │           │   │   ├── _arraysetops_impl.py
│               │           │   │   ├── _arrayterator_impl.py
│               │           │   │   ├── _datasource.py
│               │           │   │   ├── _format_impl.py
│               │           │   │   ├── _function_base_impl.py
│               │           │   │   ├── _histograms_impl.py
│               │           │   │   ├── _index_tricks_impl.py
│               │           │   │   ├── _iotools.py
│               │           │   │   ├── _nanfunctions_impl.py
│               │           │   │   ├── _npyio_impl.py
│               │           │   │   ├── _polynomial_impl.py
│               │           │   │   ├── _scimath_impl.py
│               │           │   │   ├── _shape_base_impl.py
│               │           │   │   ├── _stride_tricks_impl.py
│               │           │   │   ├── _twodim_base_impl.py
│               │           │   │   ├── _type_check_impl.py
│               │           │   │   ├── _ufunclike_impl.py
│               │           │   │   ├── _user_array_impl.py
│               │           │   │   ├── _utils_impl.py
│               │           │   │   ├── _version.py
│               │           │   │   ├── introspect.py
│               │           │   │   ├── mixins.py
│               │           │   │   ├── recfunctions.py
│               │           │   │   └── tests
│               │           │   │       ├── __init__.py
│               │           │   │       ├── test__datasource.py
│               │           │   │       ├── test__iotools.py
│               │           │   │       ├── test__version.py
│               │           │   │       ├── test_arraypad.py
│               │           │   │       ├── test_arraysetops.py
│               │           │   │       ├── test_format.py
│               │           │   │       ├── test_function_base.py
│               │           │   │       ├── test_histograms.py
│               │           │   │       ├── test_index_tricks.py
│               │           │   │       ├── test_io.py
│               │           │   │       ├── test_loadtxt.py
│               │           │   │       ├── test_mixins.py
│               │           │   │       ├── test_nanfunctions.py
│               │           │   │       ├── test_packbits.py
│               │           │   │       ├── test_polynomial.py
│               │           │   │       ├── test_recfunctions.py
│               │           │   │       ├── test_regression.py
│               │           │   │       ├── test_shape_base.py
│               │           │   │       ├── test_stride_tricks.py
│               │           │   │       ├── test_twodim_base.py
│               │           │   │       ├── test_type_check.py
│               │           │   │       ├── test_ufunclike.py
│               │           │   │       └── test_utils.py
│               │           │   ├── linalg
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── ma
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── matrixlib
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── polynomial
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── random
│               │           │   │   └── tests
│               │           │   │       ├── __init__.py
│               │           │   │       └── data
│               │           │   │           └── __init__.py
│               │           │   ├── testing
│               │           │   │   ├── _private
│               │           │   │   │   └── __init__.py
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── tests
│               │           │   │   └── __init__.py
│               │           │   └── typing
│               │           │       └── tests
│               │           │           └── __init__.py
│               │           ├── onnxruntime
│               │           │   ├── quantization
│               │           │   │   └── CalTableFlatBuffers
│               │           │   │       └── __init__.py
│               │           │   └── tools
│               │           │       ├── mobile_helpers
│               │           │       │   └── __init__.py
│               │           │       ├── ort_format_model
│               │           │       │   └── ort_flatbuffers_py
│               │           │       │       └── __init__.py
│               │           │       └── qdq_helpers
│               │           │           └── __init__.py
│               │           ├── pandas
│               │           │   ├── _libs
│               │           │   │   └── window
│               │           │   │       └── __init__.py
│               │           │   ├── core
│               │           │   │   ├── __init__.py
│               │           │   │   ├── _numba
│               │           │   │   │   └── __init__.py
│               │           │   │   ├── computation
│               │           │   │   │   └── __init__.py
│               │           │   │   ├── dtypes
│               │           │   │   │   └── __init__.py
│               │           │   │   ├── indexes
│               │           │   │   │   └── __init__.py
│               │           │   │   ├── interchange
│               │           │   │   │   └── __init__.py
│               │           │   │   ├── methods
│               │           │   │   │   └── __init__.py
│               │           │   │   ├── reshape
│               │           │   │   │   └── __init__.py
│               │           │   │   ├── sparse
│               │           │   │   │   └── __init__.py
│               │           │   │   ├── tools
│               │           │   │   │   └── __init__.py
│               │           │   │   └── util
│               │           │   │       └── __init__.py
│               │           │   ├── pyproject.toml
│               │           │   └── tests
│               │           │       ├── __init__.py
│               │           │       ├── api
│               │           │       │   └── __init__.py
│               │           │       ├── apply
│               │           │       │   └── __init__.py
│               │           │       ├── arithmetic
│               │           │       │   └── __init__.py
│               │           │       ├── arrays
│               │           │       │   ├── __init__.py
│               │           │       │   ├── boolean
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── categorical
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── datetimes
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── floating
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── integer
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── interval
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── masked
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── numpy_
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── period
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── sparse
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── string_
│               │           │       │   │   └── __init__.py
│               │           │       │   └── timedeltas
│               │           │       │       └── __init__.py
│               │           │       ├── base
│               │           │       │   └── __init__.py
│               │           │       ├── computation
│               │           │       │   └── __init__.py
│               │           │       ├── config
│               │           │       │   └── __init__.py
│               │           │       ├── construction
│               │           │       │   └── __init__.py
│               │           │       ├── copy_view
│               │           │       │   ├── __init__.py
│               │           │       │   └── index
│               │           │       │       └── __init__.py
│               │           │       ├── dtypes
│               │           │       │   ├── __init__.py
│               │           │       │   └── cast
│               │           │       │       └── __init__.py
│               │           │       ├── extension
│               │           │       │   └── __init__.py
│               │           │       ├── frame
│               │           │       │   ├── __init__.py
│               │           │       │   ├── constructors
│               │           │       │   │   └── __init__.py
│               │           │       │   └── indexing
│               │           │       │       └── __init__.py
│               │           │       ├── generic
│               │           │       │   └── __init__.py
│               │           │       ├── groupby
│               │           │       │   ├── aggregate
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── methods
│               │           │       │   │   └── __init__.py
│               │           │       │   └── transform
│               │           │       │       └── __init__.py
│               │           │       ├── indexes
│               │           │       │   ├── __init__.py
│               │           │       │   ├── base_class
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── categorical
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── datetimelike_
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── datetimes
│               │           │       │   │   ├── __init__.py
│               │           │       │   │   └── methods
│               │           │       │   │       └── __init__.py
│               │           │       │   ├── interval
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── multi
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── numeric
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── object
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── period
│               │           │       │   │   ├── __init__.py
│               │           │       │   │   └── methods
│               │           │       │   │       └── __init__.py
│               │           │       │   ├── ranges
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── string
│               │           │       │   │   └── __init__.py
│               │           │       │   └── timedeltas
│               │           │       │       ├── __init__.py
│               │           │       │       └── methods
│               │           │       │           └── __init__.py
│               │           │       ├── indexing
│               │           │       │   ├── __init__.py
│               │           │       │   ├── interval
│               │           │       │   │   └── __init__.py
│               │           │       │   └── multiindex
│               │           │       │       └── __init__.py
│               │           │       ├── interchange
│               │           │       │   └── __init__.py
│               │           │       ├── internals
│               │           │       │   └── __init__.py
│               │           │       ├── io
│               │           │       │   ├── __init__.py
│               │           │       │   ├── excel
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── formats
│               │           │       │   │   ├── __init__.py
│               │           │       │   │   └── style
│               │           │       │   │       └── __init__.py
│               │           │       │   ├── json
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── parser
│               │           │       │   │   ├── __init__.py
│               │           │       │   │   ├── common
│               │           │       │   │   │   └── __init__.py
│               │           │       │   │   ├── dtypes
│               │           │       │   │   │   └── __init__.py
│               │           │       │   │   └── usecols
│               │           │       │   │       └── __init__.py
│               │           │       │   ├── pytables
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── sas
│               │           │       │   │   └── __init__.py
│               │           │       │   └── xml
│               │           │       │       └── __init__.py
│               │           │       ├── libs
│               │           │       │   └── __init__.py
│               │           │       ├── plotting
│               │           │       │   ├── __init__.py
│               │           │       │   └── frame
│               │           │       │       └── __init__.py
│               │           │       ├── resample
│               │           │       │   └── __init__.py
│               │           │       ├── reshape
│               │           │       │   ├── __init__.py
│               │           │       │   ├── concat
│               │           │       │   │   └── __init__.py
│               │           │       │   └── merge
│               │           │       │       └── __init__.py
│               │           │       ├── scalar
│               │           │       │   ├── __init__.py
│               │           │       │   ├── interval
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── period
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── timedelta
│               │           │       │   │   ├── __init__.py
│               │           │       │   │   └── methods
│               │           │       │   │       └── __init__.py
│               │           │       │   └── timestamp
│               │           │       │       ├── __init__.py
│               │           │       │       └── methods
│               │           │       │           └── __init__.py
│               │           │       ├── series
│               │           │       │   ├── __init__.py
│               │           │       │   ├── accessors
│               │           │       │   │   └── __init__.py
│               │           │       │   └── indexing
│               │           │       │       └── __init__.py
│               │           │       ├── tools
│               │           │       │   └── __init__.py
│               │           │       ├── tseries
│               │           │       │   ├── __init__.py
│               │           │       │   ├── frequencies
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── holiday
│               │           │       │   │   └── __init__.py
│               │           │       │   └── offsets
│               │           │       │       └── __init__.py
│               │           │       ├── tslibs
│               │           │       │   └── __init__.py
│               │           │       ├── util
│               │           │       │   └── __init__.py
│               │           │       └── window
│               │           │           ├── __init__.py
│               │           │           └── moments
│               │           │               └── __init__.py
│               │           ├── pip
│               │           │   ├── _internal
│               │           │   │   ├── operations
│               │           │   │   │   ├── __init__.py
│               │           │   │   │   └── build
│               │           │   │   │       └── __init__.py
│               │           │   │   ├── resolution
│               │           │   │   │   ├── __init__.py
│               │           │   │   │   ├── legacy
│               │           │   │   │   │   └── __init__.py
│               │           │   │   │   └── resolvelib
│               │           │   │   │       └── __init__.py
│               │           │   │   └── utils
│               │           │   │       └── __init__.py
│               │           │   └── _vendor
│               │           │       └── urllib3
│               │           │           ├── contrib
│               │           │           │   ├── __init__.py
│               │           │           │   └── _securetransport
│               │           │           │       └── __init__.py
│               │           │           └── packages
│               │           │               ├── __init__.py
│               │           │               └── backports
│               │           │                   └── __init__.py
│               │           ├── pyreadline3
│               │           │   └── lineeditor
│               │           │       └── __init__.py
│               │           ├── sniffio
│               │           │   └── _tests
│               │           │       └── __init__.py
│               │           ├── sympy
│               │           │   ├── algebras
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── assumptions
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── benchmarks
│               │           │   │   └── __init__.py
│               │           │   ├── calculus
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── categories
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── codegen
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── combinatorics
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── concrete
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── core
│               │           │   │   ├── benchmarks
│               │           │   │   │   └── __init__.py
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── crypto
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── diffgeom
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── discrete
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── external
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── functions
│               │           │   │   ├── combinatorial
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   ├── elementary
│               │           │   │   │   ├── benchmarks
│               │           │   │   │   │   └── __init__.py
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   └── special
│               │           │   │       ├── benchmarks
│               │           │   │       │   └── __init__.py
│               │           │   │       └── tests
│               │           │   │           └── __init__.py
│               │           │   ├── geometry
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── holonomic
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── integrals
│               │           │   │   ├── benchmarks
│               │           │   │   │   └── __init__.py
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── interactive
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── liealgebras
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── logic
│               │           │   │   ├── algorithms
│               │           │   │   │   └── __init__.py
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── matrices
│               │           │   │   ├── benchmarks
│               │           │   │   │   └── __init__.py
│               │           │   │   ├── expressions
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── multipledispatch
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── ntheory
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── parsing
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── physics
│               │           │   │   ├── biomechanics
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   ├── continuum_mechanics
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   ├── control
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   ├── hep
│               │           │   │   │   ├── __init__.py
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   ├── mechanics
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   ├── optics
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   ├── quantum
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   ├── tests
│               │           │   │   │   └── __init__.py
│               │           │   │   ├── units
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   └── vector
│               │           │   │       └── tests
│               │           │   │           └── __init__.py
│               │           │   ├── plotting
│               │           │   │   ├── backends
│               │           │   │   │   └── __init__.py
│               │           │   │   ├── intervalmath
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   ├── pygletplot
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── polys
│               │           │   │   ├── agca
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   ├── benchmarks
│               │           │   │   │   └── __init__.py
│               │           │   │   ├── domains
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   ├── matrices
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   ├── numberfields
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── printing
│               │           │   │   ├── pretty
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── sandbox
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── series
│               │           │   │   ├── benchmarks
│               │           │   │   │   └── __init__.py
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── sets
│               │           │   │   ├── handlers
│               │           │   │   │   └── __init__.py
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── simplify
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── solvers
│               │           │   │   ├── benchmarks
│               │           │   │   │   └── __init__.py
│               │           │   │   ├── diophantine
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   ├── ode
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── stats
│               │           │   │   ├── sampling
│               │           │   │   │   ├── __init__.py
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── strategies
│               │           │   │   ├── branch
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── tensor
│               │           │   │   ├── array
│               │           │   │   │   ├── expressions
│               │           │   │   │   │   └── tests
│               │           │   │   │   │       └── __init__.py
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── testing
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── unify
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   ├── utilities
│               │           │   │   ├── _compilation
│               │           │   │   │   └── tests
│               │           │   │   │       └── __init__.py
│               │           │   │   ├── mathml
│               │           │   │   │   └── data
│               │           │   │   │       └── __init__.py
│               │           │   │   └── tests
│               │           │   │       └── __init__.py
│               │           │   └── vector
│               │           │       └── tests
│               │           │           └── __init__.py
│               │           ├── trio
│               │           │   ├── _core
│               │           │   │   └── _tests
│               │           │   │       └── __init__.py
│               │           │   ├── _tests
│               │           │   │   ├── __init__.py
│               │           │   │   └── tools
│               │           │   │       └── __init__.py
│               │           │   └── _tools
│               │           │       └── __init__.py
│               │           ├── tzdata
│               │           │   └── zoneinfo
│               │           │       ├── Africa
│               │           │       │   └── __init__.py
│               │           │       ├── America
│               │           │       │   ├── Argentina
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── Indiana
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── Kentucky
│               │           │       │   │   └── __init__.py
│               │           │       │   ├── North_Dakota
│               │           │       │   │   └── __init__.py
│               │           │       │   └── __init__.py
│               │           │       ├── Antarctica
│               │           │       │   └── __init__.py
│               │           │       ├── Arctic
│               │           │       │   └── __init__.py
│               │           │       ├── Asia
│               │           │       │   └── __init__.py
│               │           │       ├── Atlantic
│               │           │       │   └── __init__.py
│               │           │       ├── Australia
│               │           │       │   └── __init__.py
│               │           │       ├── Brazil
│               │           │       │   └── __init__.py
│               │           │       ├── Canada
│               │           │       │   └── __init__.py
│               │           │       ├── Chile
│               │           │       │   └── __init__.py
│               │           │       ├── Etc
│               │           │       │   └── __init__.py
│               │           │       ├── Europe
│               │           │       │   └── __init__.py
│               │           │       ├── Indian
│               │           │       │   └── __init__.py
│               │           │       ├── Mexico
│               │           │       │   └── __init__.py
│               │           │       ├── Pacific
│               │           │       │   └── __init__.py
│               │           │       ├── US
│               │           │       │   └── __init__.py
│               │           │       └── __init__.py
│               │           ├── urllib3
│               │           │   └── contrib
│               │           │       └── __init__.py
│               │           ├── webdriver_manager
│               │           │   ├── core
│               │           │   │   └── __init__.py
│               │           │   └── drivers
│               │           │       └── __init__.py
│               │           ├── websocket
│               │           │   └── tests
│               │           │       └── __init__.py
│               │           ├── werkzeug
│               │           │   ├── middleware
│               │           │   │   └── __init__.py
│               │           │   └── sansio
│               │           │       └── __init__.py
│               │           └── wtforms
│               │               ├── csrf
│               │               │   └── __init__.py
│               │               └── locale
│               │                   └── README.md
│               ├── app
│               │   ├── __init__.py
│               │   ├── forms.py
│               │   ├── models.py
│               │   ├── models_backup.py
│               │   ├── routes
│               │   │   ├── __init__.py
│               │   │   └── image_crawler_route.py
│               │   ├── routes.py
│               │   ├── routes_backup.py
│               │   └── utils
│               │       ├── ai_background_remover.py
│               │       ├── ai_generate_descriptions.py
│               │       ├── ai_image_crawler.py
│               │       ├── buyma_catalog_manager.py
│               │       ├── csv_handler.py
│               │       ├── models
│               │       │   └── ai_model_builder.py
│               │       ├── pricing_calculator.py
│               │       └── utils
│               │           ├── ai_background_remover.py
│               │           ├── ai_generate_descriptions.py
│               │           ├── ai_image_crawler.py
│               │           ├── buyma_catalog_manager.py
│               │           ├── csv_handler.py
│               │           ├── models
│               │           │   └── ai_model_builder.py
│               │           └── pricing_calculator.py
│               ├── project-root
│               │   ├── README.md
│               │   └── requirements.txt
│               └── requirements.txt
├── exports_atelier
│   └── atelier-kyo-manager_AIAnalysisPack(forGemini)_20250824_074807
│       └── README.md
├── project-root
│   ├── README.md
│   └── requirements.txt
└── requirements.txt
```
