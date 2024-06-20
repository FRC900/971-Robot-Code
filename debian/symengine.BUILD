load("@bazel_skylib//rules:write_file.bzl", "write_file")

genrule(
    name = "symengine_config_h_gen",
    srcs = ["symengine/symengine_config.h.in"],
    outs = ["symengine/symengine_config.h"],
    cmd = "|".join(
        [
            "cat $(SRCS)",
            "sed 's/$${SYMENGINE_MAJOR_VERSION}/0/'",
            "sed 's/$${SYMENGINE_MINOR_VERSION}/11/'",
            "sed 's/$${SYMENGINE_PATCH_VERSION}/2/'",
            "sed 's/$${SYMENGINE_VERSION}/0.11.2/'",
            "sed 's/$${SYMENGINE_INTEGER_CLASS}/GMP/'",
            "sed 's/$${SYMENGINE_SIZEOF_LONG_DOUBLE}/16/'",
        ] + [
            "sed 's/#cmakedefine " + x + "/\\/* #undef " + x + " *\\//'"
            for x in [
                "WITH_SYMENGINE_ASSERT",
                "WITH_SYMENGINE_TEUCHOS",
                "WITH_SYMENGINE_THREAD_SAFE",
                "HAVE_SYMENGINE_ECM",
                "HAVE_SYMENGINE_PRIMESIEVE",
                "WITH_SYMENGINE_VIRTUAL_TYPEID",
                "HAVE_SYMENGINE_FLINT",
                "HAVE_SYMENGINE_ARB",
                "HAVE_SYMENGINE_PIRANHA",
                "HAVE_SYMENGINE_BOOST",
                "HAVE_SYMENGINE_LLVM",
                "HAVE_C_FUNCTION_NOT_FUNC",
                # TODO(austin): Turn these on at some point...
                "HAVE_SYMENGINE_MPFR",
                "HAVE_SYMENGINE_MPC",
            ]
        ] + [
            "sed 's/#cmakedefine " + x + "/#define " + x + "/'"
            for x in [
                "HAVE_SYMENGINE_GMP",
                "WITH_SYMENGINE_RCP",
                "HAVE_SYMENGINE_PTHREAD",
                "HAVE_DEFAULT_CONSTRUCTORS",
                "HAVE_SYMENGINE_NOEXCEPT",
                "HAVE_SYMENGINE_IS_CONSTRUCTIBLE",
                "HAVE_SYMENGINE_RESERVE",
                "HAVE_SYMENGINE_STD_TO_STRING",
            ]
        ] + [
            " cat - > $(OUTS)",
        ],
    ),
)

write_file(
    name = "symengine_export_h",
    out = "symengine/symengine_export.h",
    content = [
        "",
        "#ifndef SYMENGINE_EXPORT_H",
        "#define SYMENGINE_EXPORT_H",
        "",
        "#ifdef SYMENGINE_STATIC_DEFINE",
        "#  define SYMENGINE_EXPORT",
        "#  define SYMENGINE_NO_EXPORT",
        "#else",
        "#  ifndef SYMENGINE_EXPORT",
        "#    ifdef symengine_EXPORTS",
        "        /* We are building this library */",
        "#      define SYMENGINE_EXPORT",
        "#    else",
        "        /* We are using this library */",
        "#      define SYMENGINE_EXPORT",
        "#    endif",
        "#  endif",
        "",
        "#  ifndef SYMENGINE_NO_EXPORT",
        "#    define SYMENGINE_NO_EXPORT",
        "#  endif",
        "#endif",
        "",
        "#ifndef SYMENGINE_DEPRECATED",
        "#  define SYMENGINE_DEPRECATED __attribute__ ((__deprecated__))",
        "#endif",
        "",
        "#ifndef SYMENGINE_DEPRECATED_EXPORT",
        "#  define SYMENGINE_DEPRECATED_EXPORT SYMENGINE_EXPORT SYMENGINE_DEPRECATED",
        "#endif",
        "",
        "#ifndef SYMENGINE_DEPRECATED_NO_EXPORT",
        "#  define SYMENGINE_DEPRECATED_NO_EXPORT SYMENGINE_NO_EXPORT SYMENGINE_DEPRECATED",
        "#endif",
        "",
        "#if 0 /* DEFINE_NO_DEPRECATED */",
        "#  ifndef SYMENGINE_NO_DEPRECATED",
        "#    define SYMENGINE_NO_DEPRECATED",
        "#  endif",
        "#endif",
        "",
        "#endif /* SYMENGINE_EXPORT_H */",
    ],
    is_executable = False,
)

cc_library(
    name = "cerial",
    hdrs = [
        "symengine/utilities/cereal/include/cereal/access.hpp",
        "symengine/utilities/cereal/include/cereal/archives/adapters.hpp",
        "symengine/utilities/cereal/include/cereal/archives/binary.hpp",
        "symengine/utilities/cereal/include/cereal/archives/json.hpp",
        "symengine/utilities/cereal/include/cereal/archives/portable_binary.hpp",
        "symengine/utilities/cereal/include/cereal/archives/xml.hpp",
        "symengine/utilities/cereal/include/cereal/cereal.hpp",
        "symengine/utilities/cereal/include/cereal/details/helpers.hpp",
        "symengine/utilities/cereal/include/cereal/details/polymorphic_impl.hpp",
        "symengine/utilities/cereal/include/cereal/details/polymorphic_impl_fwd.hpp",
        "symengine/utilities/cereal/include/cereal/details/static_object.hpp",
        "symengine/utilities/cereal/include/cereal/details/traits.hpp",
        "symengine/utilities/cereal/include/cereal/details/util.hpp",
        "symengine/utilities/cereal/include/cereal/macros.hpp",
        "symengine/utilities/cereal/include/cereal/specialize.hpp",
        "symengine/utilities/cereal/include/cereal/types/array.hpp",
        "symengine/utilities/cereal/include/cereal/types/atomic.hpp",
        "symengine/utilities/cereal/include/cereal/types/base_class.hpp",
        "symengine/utilities/cereal/include/cereal/types/bitset.hpp",
        "symengine/utilities/cereal/include/cereal/types/boost_variant.hpp",
        "symengine/utilities/cereal/include/cereal/types/chrono.hpp",
        "symengine/utilities/cereal/include/cereal/types/common.hpp",
        "symengine/utilities/cereal/include/cereal/types/complex.hpp",
        "symengine/utilities/cereal/include/cereal/types/concepts/pair_associative_container.hpp",
        "symengine/utilities/cereal/include/cereal/types/deque.hpp",
        "symengine/utilities/cereal/include/cereal/types/forward_list.hpp",
        "symengine/utilities/cereal/include/cereal/types/functional.hpp",
        "symengine/utilities/cereal/include/cereal/types/list.hpp",
        "symengine/utilities/cereal/include/cereal/types/map.hpp",
        "symengine/utilities/cereal/include/cereal/types/memory.hpp",
        "symengine/utilities/cereal/include/cereal/types/optional.hpp",
        "symengine/utilities/cereal/include/cereal/types/polymorphic.hpp",
        "symengine/utilities/cereal/include/cereal/types/queue.hpp",
        "symengine/utilities/cereal/include/cereal/types/set.hpp",
        "symengine/utilities/cereal/include/cereal/types/stack.hpp",
        "symengine/utilities/cereal/include/cereal/types/string.hpp",
        "symengine/utilities/cereal/include/cereal/types/tuple.hpp",
        "symengine/utilities/cereal/include/cereal/types/unordered_map.hpp",
        "symengine/utilities/cereal/include/cereal/types/unordered_set.hpp",
        "symengine/utilities/cereal/include/cereal/types/utility.hpp",
        "symengine/utilities/cereal/include/cereal/types/valarray.hpp",
        "symengine/utilities/cereal/include/cereal/types/variant.hpp",
        "symengine/utilities/cereal/include/cereal/types/vector.hpp",
        "symengine/utilities/cereal/include/cereal/version.hpp",
    ],
    includes = [
        "symengine/utilities/cereal/include",
    ],
)

cc_library(
    name = "symengine",
    srcs = [
        "symengine/add.cpp",
        "symengine/assumptions.cpp",
        "symengine/basic.cpp",
        "symengine/complex.cpp",
        "symengine/complex_double.cpp",
        "symengine/constants.cpp",
        "symengine/cse.cpp",
        "symengine/cwrapper.cpp",
        "symengine/dense_matrix.cpp",
        "symengine/derivative.cpp",
        "symengine/dict.cpp",
        "symengine/diophantine.cpp",
        "symengine/eval.cpp",
        "symengine/eval_double.cpp",
        "symengine/expand.cpp",
        "symengine/expression.cpp",
        "symengine/fields.cpp",
        "symengine/finitediff.cpp",
        "symengine/functions.cpp",
        "symengine/infinity.cpp",
        "symengine/integer.cpp",
        "symengine/logic.cpp",
        "symengine/matrices/conjugate_matrix.cpp",
        "symengine/matrices/diagonal_matrix.cpp",
        "symengine/matrices/hadamard_product.cpp",
        "symengine/matrices/identity_matrix.cpp",
        "symengine/matrices/immutable_dense_matrix.cpp",
        "symengine/matrices/is_diagonal.cpp",
        "symengine/matrices/is_lower.cpp",
        "symengine/matrices/is_real.cpp",
        "symengine/matrices/is_square.cpp",
        "symengine/matrices/is_symmetric.cpp",
        "symengine/matrices/is_toeplitz.cpp",
        "symengine/matrices/is_upper.cpp",
        "symengine/matrices/is_zero.cpp",
        "symengine/matrices/matrix_add.cpp",
        "symengine/matrices/matrix_mul.cpp",
        "symengine/matrices/matrix_symbol.cpp",
        "symengine/matrices/size.cpp",
        "symengine/matrices/trace.cpp",
        "symengine/matrices/transpose.cpp",
        "symengine/matrices/zero_matrix.cpp",
        "symengine/matrix.cpp",
        "symengine/monomials.cpp",
        "symengine/mp_wrapper.cpp",
        "symengine/mul.cpp",
        "symengine/nan.cpp",
        "symengine/ntheory.cpp",
        "symengine/ntheory_funcs.cpp",
        "symengine/number.cpp",
        "symengine/numer_denom.cpp",
        "symengine/parser/parser.cpp",
        "symengine/parser/parser.tab.cc",
        "symengine/parser/parser_old.cpp",
        "symengine/parser/sbml/sbml_parser.cpp",
        "symengine/parser/sbml/sbml_parser.tab.cc",
        "symengine/parser/sbml/sbml_tokenizer.cpp",
        "symengine/parser/tokenizer.cpp",
        "symengine/polys/basic_conversions.cpp",
        "symengine/polys/msymenginepoly.cpp",
        "symengine/polys/uexprpoly.cpp",
        "symengine/polys/uintpoly.cpp",
        "symengine/polys/uratpoly.cpp",
        "symengine/pow.cpp",
        "symengine/prime_sieve.cpp",
        "symengine/printers/codegen.cpp",
        "symengine/printers/latex.cpp",
        "symengine/printers/mathml.cpp",
        "symengine/printers/sbml.cpp",
        "symengine/printers/stringbox.cpp",
        "symengine/printers/strprinter.cpp",
        "symengine/printers/unicode.cpp",
        "symengine/rational.cpp",
        "symengine/real_double.cpp",
        "symengine/refine.cpp",
        "symengine/rewrite.cpp",
        "symengine/rings.cpp",
        "symengine/series.cpp",
        "symengine/series_generic.cpp",
        "symengine/set_funcs.cpp",
        "symengine/sets.cpp",
        "symengine/simplify.cpp",
        "symengine/solve.cpp",
        "symengine/sparse_matrix.cpp",
        "symengine/symbol.cpp",
        "symengine/symengine_rcp.cpp",
        "symengine/test_visitors.cpp",
        "symengine/tuple.cpp",
        "symengine/visitor.cpp",
    ],
    hdrs = [
        "symengine/add.h",
        "symengine/assumptions.h",
        "symengine/basic.h",
        "symengine/basic-inl.h",
        "symengine/basic-methods.inc",
        "symengine/complex.h",
        "symengine/complex_double.h",
        "symengine/complex_mpc.h",
        "symengine/constants.h",
        "symengine/cwrapper.h",
        "symengine/derivative.h",
        "symengine/dict.h",
        "symengine/diophantine.h",
        "symengine/eval.h",
        "symengine/eval_double.h",
        "symengine/eval_mpc.h",
        "symengine/eval_mpfr.h",
        "symengine/expression.h",
        "symengine/fields.h",
        "symengine/finitediff.h",
        "symengine/functions.h",
        "symengine/infinity.h",
        "symengine/integer.h",
        "symengine/lambda_double.h",
        "symengine/logic.h",
        "symengine/matrices/conjugate_matrix.h",
        "symengine/matrices/diagonal_matrix.h",
        "symengine/matrices/hadamard_product.h",
        "symengine/matrices/identity_matrix.h",
        "symengine/matrices/immutable_dense_matrix.h",
        "symengine/matrices/matrix_add.h",
        "symengine/matrices/matrix_expr.h",
        "symengine/matrices/matrix_mul.h",
        "symengine/matrices/matrix_symbol.h",
        "symengine/matrices/size.h",
        "symengine/matrices/trace.h",
        "symengine/matrices/transpose.h",
        "symengine/matrices/zero_matrix.h",
        "symengine/matrix.h",
        "symengine/matrix_expressions.h",
        "symengine/monomials.h",
        "symengine/mp_class.h",
        "symengine/mp_wrapper.h",
        "symengine/mul.h",
        "symengine/nan.h",
        "symengine/ntheory.h",
        "symengine/ntheory_funcs.h",
        "symengine/number.h",
        "symengine/parser.h",
        "symengine/parser/parser.h",
        "symengine/parser/parser.tab.hh",
        "symengine/parser/sbml/sbml_parser.h",
        "symengine/parser/sbml/sbml_parser.tab.hh",
        "symengine/parser/sbml/sbml_tokenizer.h",
        "symengine/parser/tokenizer.h",
        "symengine/polys/basic_conversions.h",
        "symengine/polys/msymenginepoly.h",
        "symengine/polys/uexprpoly.h",
        "symengine/polys/uintpoly.h",
        "symengine/polys/uintpoly_flint.h",
        "symengine/polys/uintpoly_piranha.h",
        "symengine/polys/upolybase.h",
        "symengine/polys/uratpoly.h",
        "symengine/polys/usymenginepoly.h",
        "symengine/pow.h",
        "symengine/prime_sieve.h",
        "symengine/printers.h",
        "symengine/printers/codegen.h",
        "symengine/printers/latex.h",
        "symengine/printers/mathml.h",
        "symengine/printers/sbml.h",
        "symengine/printers/stringbox.h",
        "symengine/printers/strprinter.h",
        "symengine/printers/unicode.h",
        "symengine/rational.h",
        "symengine/real_double.h",
        "symengine/real_mpfr.h",
        "symengine/refine.h",
        "symengine/rings.h",
        "symengine/serialize-cereal.h",
        "symengine/series.h",
        "symengine/series_flint.h",
        "symengine/series_generic.h",
        "symengine/series_piranha.h",
        "symengine/series_visitor.h",
        "symengine/sets.h",
        "symengine/simplify.h",
        "symengine/solve.h",
        "symengine/subs.h",
        "symengine/symbol.h",
        "symengine/symengine_assert.h",
        "symengine/symengine_casts.h",
        "symengine/symengine_exception.h",
        "symengine/symengine_rcp.h",
        "symengine/test_visitors.h",
        "symengine/tribool.h",
        "symengine/tuple.h",
        "symengine/type_codes.inc",
        "symengine/utilities/stream_fmt.h",
        "symengine/visitor.h",
    ],
    copts = ["-Wno-unused-but-set-variable"],
    includes = [
        ".",
    ],
    textual_hdrs = [
        "symengine/as_real_imag.cpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cerial",
        ":config",
        "@//third_party/gmp",
    ],
)

cc_library(
    name = "config",
    hdrs = [
        "symengine/symengine_config.h",
        "symengine/symengine_export.h",
    ],
    includes = [
        ".",
    ],
)

cc_library(
    name = "catch",
    srcs = [
        "symengine/utilities/catch/catch.cpp",
    ],
    hdrs = [
        "symengine/utilities/catch/catch.hpp",
    ],
    includes = [
        "symengine/utilities/catch",
    ],
    textual_hdrs = [
        "symengine/numer_denom.cpp",
    ],
    deps = [":symengine"],
)

[
    cc_test(
        name = x,
        srcs = [
            "symengine/tests/basic/" + x + ".cpp",
        ],
        deps = [
            ":catch",
            ":symengine",
        ],
    )
    for x in [
        "test_basic",
        "test_arit",
        "test_poly",
        "test_series",
        "test_series_generic",
        "test_functions",
        "test_subs",
        "test_integer",
        "test_rational",
        "test_relationals",
        "test_number",
        "test_as_numer_denom",
        "test_parser",
        "test_serialize-cereal",
        "test_sbml_parser",
        "test_sets",
        "test_fields",
        "test_infinity",
        "test_nan",
        "test_solve",
        "test_as_real_imag",
        "test_cse",
        "test_count_ops",
        "test_test_visitors",
        "test_assumptions",
        "test_refine",
        "test_simplify",
        "test_tuple",
        "test_tribool",
    ]
]

[
    cc_test(
        name = x,
        srcs = [
            "symengine/tests/eval/" + x + ".cpp",
        ],
        deps = [
            ":catch",
            ":symengine",
        ],
    )
    for x in [
        "test_evalf",
        "test_eval_double",
        "test_lambda_double",
    ]
]

cc_test(
    name = "test_expression",
    srcs = [
        "symengine/tests/expression/test_expression.cpp",
    ],
    deps = [
        ":catch",
        ":symengine",
    ],
)

cc_test(
    name = "test_finitediff",
    srcs = [
        "symengine/tests/finitediff/test_finitediff.cpp",
    ],
    deps = [
        ":catch",
        ":symengine",
    ],
)

cc_test(
    name = "test_logic",
    srcs = [
        "symengine/tests/logic/test_logic.cpp",
    ],
    deps = [
        ":catch",
        ":symengine",
    ],
)

[
    cc_test(
        name = x,
        srcs = [
            "symengine/tests/matrix/" + x + ".cpp",
        ],
        deps = [
            ":catch",
            ":symengine",
        ],
    )
    for x in [
        "test_matrix",
        "test_matrixexpr",
    ]
]

[
    cc_test(
        name = x,
        srcs = [
            "symengine/tests/ntheory/" + x + ".cpp",
        ],
        deps = [
            ":catch",
            ":symengine",
        ],
    )
    for x in [
        "test_ntheory",
        "test_diophantine",
        "test_ntheory_funcs",
    ]
]

[
    cc_test(
        name = x,
        srcs = [
            "symengine/tests/polynomial/" + x + ".cpp",
        ],
        deps = [
            ":catch",
            ":symengine",
        ],
    )
    for x in [
        "test_uintpoly",
        "test_uratpoly",
        "test_mintpoly",
        "test_uexprpoly",
        "test_mexprpoly",
        "test_basic_conversions",
    ]
]

[
    cc_test(
        name = x,
        srcs = [
            "symengine/tests/printing/" + x + ".cpp",
        ],
        deps = [
            ":catch",
            ":symengine",
        ],
    )
    for x in [
        "test_printing",
        "test_ccode",
    ]
]

cc_test(
    name = "test_rcp",
    srcs = [
        "symengine/tests/rcp/test_rcp.cpp",
    ],
    deps = [
        ":catch",
        ":symengine",
    ],
)
