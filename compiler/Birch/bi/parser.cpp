/* A Bison parser, made by GNU Bison 3.5.4.  */

/* Skeleton implementation for Bison GLR parsers in C

   Copyright (C) 2002-2015, 2018-2020 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C GLR parser skeleton written by Paul Hilfinger.  */

/* Undocumented macros, especially those whose name start with YY_,
   are private implementation details.  Do not rely on them.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.5.4"

/* Skeleton name.  */
#define YYSKELETON_NAME "glr.c"

/* Pure parsers.  */
#define YYPURE 0







# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

#include "bi/parser.hpp"

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Default (constant) value used for initialization for null
   right-hand sides.  Unlike the standard yacc.c template, here we set
   the default value of $$ to a zeroed-out value.  Since the default
   value is undefined, this behavior is technically correct.  */
static YYSTYPE yyval_default;
static YYLTYPE yyloc_default
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
  = { 1, 1, 1, 1 }
# endif
;


/* Unqualified %code blocks.  */
#line 6 "bi/parser.ypp"

  #include "bi/expression/all.hpp"
  #include "bi/statement/all.hpp"
  #include "bi/type/all.hpp"

  /**
   * Raw string stack.
   */
  std::stack<std::string> raws;

  /**
   * Push the current raw string onto the stack, and restart it.
   */
  void push_raw() {
    raws.push(raw.str());
    raw.str("");
  }

  /**
   * Pop a raw string from the stack.
   */
  std::string pop_raw() {
    std::string raw = raws.top();
    raws.pop();
    return raw;
  }

  /**
   * Make a location, without documentation string.
   */
  bi::Location* make_loc(YYLTYPE& loc) {
    return new bi::Location(compiler->file, loc.first_line, loc.last_line,
        loc.first_column, loc.last_column);
  }

  /**
   * Make a location, with documentation string.
   */
  bi::Location* make_doc_loc(YYLTYPE& loc) {
    return new bi::Location(compiler->file, loc.first_line, loc.last_line,
        loc.first_column, loc.last_column, pop_raw());
  }

  /**
   * Make an empty expression.
   */
  bi::Expression* empty_expr(YYLTYPE& loc) {
    return new bi::EmptyExpression(make_loc(loc));
  }

  /**
   * Make an empty statement.
   */
  bi::Statement* empty_stmt(YYLTYPE& loc) {
    return new bi::EmptyStatement(make_loc(loc));
  }

  /**
   * Make an empty type.
   */
  bi::Type* empty_type(YYLTYPE& loc) {
    return new bi::EmptyType(make_loc(loc));
  }

#line 166 "bi/parser.cpp"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef YYFREE
# define YYFREE free
#endif
#ifndef YYMALLOC
# define YYMALLOC malloc
#endif
#ifndef YYREALLOC
# define YYREALLOC realloc
#endif

#define YYSIZEMAX \
  (PTRDIFF_MAX < SIZE_MAX ? PTRDIFF_MAX : YY_CAST (ptrdiff_t, SIZE_MAX))

#ifdef __cplusplus
  typedef bool yybool;
# define yytrue true
# define yyfalse false
#else
  /* When we move to stdbool, get rid of the various casts to yybool.  */
  typedef signed char yybool;
# define yytrue 1
# define yyfalse 0
#endif

#ifndef YYSETJMP
# include <setjmp.h>
# define YYJMP_BUF jmp_buf
# define YYSETJMP(Env) setjmp (Env)
/* Pacify Clang and ICC.  */
# define YYLONGJMP(Env, Val)                    \
 do {                                           \
   longjmp (Env, Val);                          \
   YY_ASSERT (0);                               \
 } while (yyfalse)
#endif

#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* The _Noreturn keyword of C11.  */
#ifndef _Noreturn
# if (defined __cplusplus \
      && ((201103 <= __cplusplus && !(__GNUC__ == 4 && __GNUC_MINOR__ == 7)) \
          || (defined _MSC_VER && 1900 <= _MSC_VER)))
#  define _Noreturn [[noreturn]]
# elif ((!defined __cplusplus || defined __clang__) \
        && (201112 <= (defined __STDC_VERSION__ ? __STDC_VERSION__ : 0)  \
            || 4 < __GNUC__ + (7 <= __GNUC_MINOR__)))
   /* _Noreturn works as-is.  */
# elif 2 < __GNUC__ + (8 <= __GNUC_MINOR__) || 0x5110 <= __SUNPRO_C
#  define _Noreturn __attribute__ ((__noreturn__))
# elif 1200 <= (defined _MSC_VER ? _MSC_VER : 0)
#  define _Noreturn __declspec (noreturn)
# else
#  define _Noreturn
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && ! defined __ICC && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                            \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  38
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   628

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  74
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  135
/* YYNRULES -- Number of rules.  */
#define YYNRULES  275
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  471
/* YYMAXRHS -- Maximum number of symbols on right-hand side of rule.  */
#define YYMAXRHS 10
/* YYMAXLEFT -- Maximum number of symbols to the left of a handle
   accessed by $0, $-1, etc., in any rule.  */
#define YYMAXLEFT 0

/* YYMAXUTOK -- Last valid token number (for yychar).  */
#define YYMAXUTOK   303
/* YYUNDEFTOK -- Symbol number (for yytoken) that denotes an unknown
   token.  */
#define YYUNDEFTOK  2

/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                         \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    57,     2,     2,     2,    58,    73,     2,
      49,    50,    62,    59,    55,    60,    56,    63,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    66,    68,
      64,    69,    65,    53,    61,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    51,    54,    52,     2,    67,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    71,     2,    72,    70,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48
};

#if YYDEBUG
/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   159,   159,   168,   172,   176,   180,   184,   185,   186,
     187,   191,   195,   199,   203,   207,   211,   215,   219,   223,
     224,   225,   226,   227,   228,   229,   230,   234,   235,   239,
     240,   244,   248,   249,   250,   251,   252,   253,   254,   255,
     261,   262,   266,   267,   268,   272,   273,   274,   278,   279,
     283,   284,   288,   289,   293,   294,   298,   299,   300,   301,
     310,   311,   315,   316,   320,   321,   325,   329,   330,   334,
     338,   339,   343,   347,   348,   352,   356,   357,   361,   362,
     366,   370,   371,   375,   379,   380,   384,   385,   389,   390,
     394,   398,   399,   403,   404,   408,   412,   413,   417,   418,
     422,   423,   427,   428,   432,   433,   437,   441,   442,   446,
     447,   451,   452,   456,   460,   461,   470,   471,   472,   473,
     474,   478,   479,   480,   481,   482,   486,   487,   488,   489,
     490,   491,   495,   495,   499,   499,   503,   504,   505,   506,
     512,   512,   513,   513,   517,   517,   518,   518,   522,   522,
     526,   527,   528,   529,   530,   531,   535,   539,   539,   543,
     543,   547,   547,   551,   551,   555,   556,   557,   561,   561,
     562,   562,   563,   563,   567,   568,   569,   573,   577,   581,
     582,   583,   584,   588,   592,   596,   597,   598,   602,   606,
     610,   616,   617,   621,   625,   629,   633,   637,   641,   645,
     646,   647,   648,   649,   650,   651,   652,   653,   654,   655,
     656,   657,   661,   662,   666,   667,   671,   672,   673,   674,
     675,   676,   680,   681,   685,   686,   690,   691,   692,   693,
     694,   695,   696,   697,   698,   699,   703,   704,   708,   709,
     713,   717,   721,   722,   726,   727,   728,   732,   733,   737,
     741,   742,   746,   750,   751,   755,   759,   760,   764,   773,
     774,   778,   779,   783,   784,   785,   786,   787,   788,   789,
     790,   791,   795,   796,   800,   801
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "PROGRAM", "CLASS", "TYPE", "FUNCTION",
  "FIBER", "OPERATOR", "AUTO", "IF", "ELSE", "FOR", "IN", "WHILE", "DO",
  "ASSERT", "RETURN", "YIELD", "CPP", "HPP", "THIS", "SUPER", "GLOBAL",
  "PARALLEL", "DYNAMIC", "ABSTRACT", "OVERRIDE", "FINAL", "NIL",
  "DOUBLE_BRACE_OPEN", "DOUBLE_BRACE_CLOSE", "NAME", "BOOL_LITERAL",
  "INT_LITERAL", "REAL_LITERAL", "STRING_LITERAL", "LEFT_OP", "RIGHT_OP",
  "LEFT_TILDE_OP", "RIGHT_TILDE_OP", "LEFT_QUERY_OP", "AND_OP", "OR_OP",
  "LE_OP", "GE_OP", "EQ_OP", "NE_OP", "RANGE_OP", "'('", "')'", "'['",
  "']'", "'?'", "'\\\\'", "','", "'.'", "'!'", "'%'", "'+'", "'-'", "'@'",
  "'*'", "'/'", "'<'", "'>'", "':'", "'_'", "';'", "'='", "'~'", "'{'",
  "'}'", "'&'", "$accept", "name", "bool_literal", "int_literal",
  "real_literal", "string_literal", "literal", "identifier",
  "parens_expression", "sequence_expression", "cast_expression",
  "function_expression", "this_expression", "super_expression",
  "nil_expression", "primary_expression", "index_expression", "index_list",
  "slice", "postfix_expression", "query_expression", "prefix_operator",
  "prefix_expression", "multiplicative_operator",
  "multiplicative_expression", "additive_operator", "additive_expression",
  "relational_operator", "relational_expression", "equality_operator",
  "equality_expression", "logical_and_operator", "logical_and_expression",
  "logical_or_operator", "logical_or_expression", "assign_operator",
  "assign_expression", "expression", "optional_expression",
  "expression_list", "span_expression", "span_list", "brackets",
  "parameters", "optional_parameters", "parameter_list", "parameter",
  "options", "option_list", "option", "arguments", "optional_arguments",
  "shape", "generics", "generic_list", "generic", "optional_generics",
  "generic_arguments", "generic_argument_list", "generic_argument",
  "optional_generic_arguments", "global_variable_declaration",
  "member_variable_declaration", "local_variable_declaration",
  "function_declaration", "$@1", "fiber_declaration", "$@2",
  "member_function_annotation", "member_function_declaration", "$@3",
  "$@4", "member_fiber_declaration", "$@5", "$@6", "program_declaration",
  "$@7", "binary_operator", "unary_operator",
  "binary_operator_declaration", "$@8", "unary_operator_declaration",
  "$@9", "assignment_operator_declaration", "$@10",
  "conversion_operator_declaration", "$@11", "class_annotation",
  "class_declaration", "$@12", "$@13", "$@14", "basic_declaration", "cpp",
  "hpp", "assume_operator", "assume_statement", "expression_statement",
  "if", "for_variable_declaration", "for", "parallel_annotation",
  "parallel", "while", "do_while", "block", "assertion", "return", "yield",
  "statement", "statements", "optional_statements", "class_statement",
  "class_statements", "optional_class_statements", "file_statement",
  "file_statements", "optional_file_statements", "file", "return_type",
  "optional_return_type", "fiber_return_type",
  "optional_fiber_return_type", "value", "optional_value", "braces",
  "optional_braces", "class_braces", "optional_class_braces",
  "double_braces", "named_type", "primary_type", "type", "type_list",
  "optional_type_list", YY_NULLPTR
};
#endif

#define YYPACT_NINF (-399)
#define YYTABLE_NINF (-240)

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     140,    17,    17,    17,    17,    43,     2,     2,  -399,  -399,
    -399,    24,  -399,  -399,  -399,  -399,  -399,  -399,    64,  -399,
    -399,  -399,  -399,   592,  -399,  -399,   100,    58,    94,    54,
      54,     5,    97,  -399,  -399,   135,    17,  -399,  -399,    48,
    -399,    17,  -399,    17,     8,  -399,    76,    76,  -399,  -399,
    -399,    69,  -399,   563,    17,  -399,   135,   103,  -399,    90,
     108,  -399,   128,   107,  -399,    95,   132,   123,    41,   119,
     143,  -399,  -399,   149,   161,    49,   180,   182,   135,  -399,
    -399,  -399,  -399,  -399,  -399,  -399,  -399,  -399,  -399,  -399,
    -399,  -399,  -399,  -399,  -399,  -399,  -399,    17,   173,   203,
     175,   135,    -2,  -399,   153,    17,   530,   489,   448,  -399,
    -399,    10,  -399,    23,   159,   160,   -16,   135,  -399,    17,
    -399,   407,  -399,  -399,  -399,  -399,  -399,    17,  -399,   183,
     179,   135,  -399,  -399,   135,  -399,  -399,   216,   185,   180,
     170,   135,  -399,  -399,   190,  -399,   186,   189,   216,  -399,
    -399,  -399,  -399,   201,  -399,  -399,  -399,  -399,  -399,   530,
     530,    76,   530,   184,  -399,  -399,  -399,  -399,  -399,  -399,
    -399,  -399,  -399,  -399,  -399,   207,  -399,  -399,   146,  -399,
     530,  -399,    75,   115,   105,   144,   222,    73,  -399,  -399,
    -399,   211,   221,   220,  -399,   223,   224,   227,  -399,  -399,
     226,  -399,   212,  -399,  -399,    17,  -399,   217,   192,  -399,
      17,   530,    17,   530,   225,   530,   530,   530,   276,  -399,
      19,    74,  -399,  -399,  -399,  -399,  -399,  -399,   268,  -399,
    -399,  -399,  -399,  -399,  -399,  -399,   407,  -399,   232,  -399,
    -399,  -399,    17,   258,    41,   266,    41,   180,  -399,  -399,
     180,  -399,   135,    17,   244,   245,   180,  -399,   257,    17,
     530,  -399,    17,  -399,  -399,  -399,  -399,  -399,   530,   530,
     530,   530,   530,  -399,   530,   530,   530,  -399,   170,   530,
    -399,  -399,  -399,  -399,  -399,    17,    52,  -399,  -399,   261,
     225,  -399,   301,   225,   299,   254,  -399,   263,   265,    17,
     135,  -399,  -399,  -399,  -399,  -399,   530,   313,  -399,  -399,
    -399,  -399,    10,  -399,  -399,    41,  -399,  -399,   270,  -399,
    -399,  -399,    41,   286,  -399,   282,   287,   292,  -399,  -399,
      75,   115,   105,   144,   222,  -399,  -399,  -399,  -399,   275,
     295,  -399,   300,  -399,  -399,   277,   335,   530,  -399,   530,
    -399,  -399,  -399,   334,   164,   281,    17,   226,    41,  -399,
    -399,  -399,   530,   530,  -399,   530,  -399,  -399,  -399,    17,
      17,   156,  -399,  -399,   323,   285,  -399,   200,  -399,  -399,
    -399,  -399,  -399,   300,  -399,   280,  -399,    11,   305,   289,
     530,  -399,    27,   290,   291,  -399,   342,  -399,   310,  -399,
    -399,    52,    54,    54,    17,  -399,  -399,   135,    17,    17,
    -399,  -399,  -399,  -399,   530,  -399,   314,  -399,   296,  -399,
    -399,   530,  -399,  -399,    76,    76,  -399,    41,   233,    54,
      54,   225,   530,  -399,   315,   180,   182,    41,  -399,  -399,
      28,   297,   298,    76,    76,  -399,   225,   530,  -399,  -399,
    -399,  -399,   303,  -399,  -399,   180,   182,  -399,   225,    41,
      41,  -399,  -399,  -399,  -399,  -399,  -399,    41,    41,  -399,
    -399
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_int16 yydefact[] =
{
     167,     0,     0,     0,     0,     0,     0,     0,   166,   165,
       2,     0,   226,   227,   228,   229,   230,   231,     0,   232,
     233,   234,   235,   167,   238,   240,     0,     0,     0,   108,
     108,     0,     0,   177,   178,     0,     0,   237,     1,     0,
     148,     0,   176,     0,     0,   107,     0,     0,    44,    42,
      43,     0,   156,     0,     0,   258,     0,     0,   268,   115,
     261,   263,     0,   108,    91,     0,     0,    93,     0,     0,
       0,   102,   106,     0,   104,     0,   243,   248,     0,    66,
      69,    58,    59,    62,    63,    52,    53,    48,    49,    56,
      57,   150,   151,   152,   153,   154,   155,     0,     0,   272,
       0,   275,     0,   114,   259,     0,     0,     0,     0,   264,
     266,     0,   116,     0,     0,     0,    87,     0,    92,     0,
     254,   215,   253,   149,   174,   175,   103,     0,    84,     0,
      88,     0,   242,   132,     0,   247,   134,    90,     0,   243,
       0,     0,   262,   274,     0,   109,     0,   111,   113,   260,
     269,    16,    17,     0,    18,     3,     4,     5,     6,     0,
       0,     0,     0,   115,     7,     8,     9,    10,    19,    20,
      21,    22,    23,    24,    25,     0,    26,    32,    40,    45,
       0,    50,    54,    60,    64,    67,    70,    73,    75,   249,
      96,    78,     0,   100,    80,    81,     0,     0,   267,   261,
       0,   119,     0,   117,   118,     0,    86,   170,   251,    94,
       0,     0,     0,     0,     0,     0,    77,     0,     0,   190,
     115,     0,   199,   211,   201,   200,   202,   203,     0,   204,
     205,   206,   207,   208,   209,   210,   212,   214,     0,   195,
     105,    85,     0,   241,     0,   244,     0,   243,   159,   273,
     243,   110,     0,     0,     0,     0,   243,    47,    11,     0,
       0,    41,     0,    38,    39,    36,    37,    46,     0,     0,
       0,     0,     0,    72,     0,     0,     0,    97,     0,     0,
      83,   270,   265,   120,   172,     0,     0,   250,    95,     0,
       0,   188,     0,     0,     0,     0,    76,     0,     0,     0,
       0,   180,   181,   182,   184,   179,     0,     0,   213,   252,
      89,   133,   246,   135,   157,     0,   271,   112,   115,    34,
      12,    13,     0,     0,    33,    29,     0,    28,    35,    51,
      55,    61,    65,    68,    71,    74,    79,   101,    82,     0,
      99,   257,   225,   256,   171,     0,   187,     0,   193,     0,
     196,   197,   198,     0,     0,     0,     0,   245,     0,   160,
      11,    15,     0,     0,    31,     0,   173,    98,   168,     0,
       0,     0,   136,   138,   137,     0,   216,     0,   217,   218,
     219,   220,   221,   222,   224,     0,   126,     0,     0,     0,
       0,   127,     0,     0,     0,   183,     0,   158,     0,    30,
      27,     0,   108,   108,     0,   163,   139,     0,     0,     0,
     223,   255,   186,   185,     0,   194,     0,   130,     0,   128,
     129,     0,    14,   169,     0,     0,   161,     0,     0,   108,
     108,     0,     0,   131,     0,   243,   248,     0,   164,   121,
       0,     0,     0,     0,     0,   189,     0,     0,   142,   146,
     162,   124,     0,   122,   123,   243,   248,   192,     0,     0,
       0,   125,   140,   144,   191,   143,   147,     0,     0,   141,
     145
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -399,     0,  -399,  -399,  -399,  -399,  -399,  -202,  -399,  -399,
    -399,  -399,  -399,  -399,  -399,  -399,  -399,     4,  -399,  -399,
    -399,   337,  -146,   320,   106,   321,   110,   324,   111,   328,
     104,   333,   114,   202,  -399,  -399,   116,   -60,  -399,  -140,
    -399,   113,  -344,   -29,  -399,   148,   -26,  -399,   274,  -399,
     -99,  -399,   120,  -399,   267,  -399,   -18,  -399,   145,  -399,
     -53,  -399,  -399,  -399,  -399,  -399,  -399,  -399,  -399,  -399,
    -399,  -399,  -399,  -399,  -399,  -399,  -399,  -399,  -399,  -399,
    -399,  -399,  -399,  -399,  -399,  -399,  -399,  -399,  -399,  -399,
    -399,  -399,  -399,  -112,  -317,  -399,  -399,  -399,    12,  -286,
    -399,  -399,  -399,  -399,  -399,  -399,  -399,  -399,  -399,  -399,
     165,  -399,  -399,    20,  -399,  -399,   377,  -399,  -399,    31,
    -117,  -399,  -398,  -181,  -399,  -110,  -229,  -399,     9,   398,
     -17,  -103,   -28,   -72,  -399
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,   163,   164,   165,   166,   167,   168,   169,   170,   171,
     172,   173,   174,   175,   176,   177,   325,   326,   265,   178,
     179,   180,   181,   268,   182,   269,   183,   270,   184,   271,
     185,   272,   186,    96,   187,   275,   188,   191,   297,   192,
     195,   196,   113,    76,   207,   129,   130,    40,    66,    67,
     114,   368,   197,    45,    73,    74,    46,   103,   146,   147,
     258,    12,   376,   222,    13,   244,    14,   246,   377,   378,
     467,   459,   379,   468,   460,    15,    68,    97,    54,    16,
     358,    17,   315,   380,   437,   381,   427,    18,    19,   401,
     286,   339,    20,    21,    22,   306,   224,   225,   226,   292,
     227,   228,   229,   230,   231,   232,   233,   234,   235,   236,
     237,   238,   383,   384,   385,    23,    24,    25,    26,   132,
     133,   135,   136,   115,   288,   122,   123,   343,   344,    33,
      60,    61,    99,   100,   144
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      11,    27,    28,    29,    30,    53,   104,    62,   200,   223,
     392,   239,    47,   353,   202,   311,   257,   313,    77,   254,
     255,   211,   248,    11,    69,   382,    70,   287,    98,   143,
      10,    51,    32,    75,   267,    59,    63,    10,   449,    65,
      10,    59,    10,    59,    72,   116,   189,    56,   194,    10,
     137,   319,    57,   205,    51,    58,    59,   324,   463,    56,
     328,   221,    48,   145,    49,    50,   382,   198,    36,   249,
     396,   138,   107,    71,   148,    51,   107,   107,    59,   266,
      10,    10,   121,   102,   440,   300,   359,   206,   150,   208,
      35,   201,    31,   361,   199,   417,   451,    51,    64,   128,
      38,    59,    59,   243,   294,    59,   245,    39,   345,   120,
     273,    59,   121,   301,   302,   303,    80,    59,    44,    65,
     341,   220,   329,   342,   223,    75,   239,    72,    55,   397,
     314,    59,   256,   316,    59,    78,   336,    87,    88,   322,
    -239,    59,   304,     1,   305,     2,     3,     4,     5,    81,
      82,   290,   101,   293,   102,   295,   296,   298,    41,     6,
       7,   117,    42,    43,   105,   106,     8,    10,     9,    89,
      90,    44,    10,   394,    85,    86,   221,   107,   119,   108,
     346,   109,   118,   348,    56,   110,   111,   124,   284,    57,
      83,    84,    58,   404,   131,   107,   112,   260,   438,   261,
     327,   106,   262,   263,   264,    59,   408,   409,   450,   357,
     289,   125,   291,   107,   126,   108,   127,   109,   131,   194,
     134,   110,   111,   139,   148,   142,   149,   203,   204,   106,
     465,   466,   391,   241,   242,   247,   220,   193,   469,   470,
     250,   367,    51,   140,   252,   109,   355,   442,   102,   110,
     111,   251,    59,   318,   140,   393,   109,   253,   141,   318,
     110,   111,   318,   259,    79,   360,   276,   140,   340,   109,
     106,   277,   354,   110,   111,   278,   280,   413,   279,   281,
     283,   285,   107,   282,   108,    59,   109,   388,   299,   389,
     110,   111,   307,   418,   320,   199,   121,   321,   106,   291,
      59,   439,   398,   327,   309,   400,   369,   370,   371,   140,
     323,   109,    59,   349,   347,   110,   111,   140,   448,   109,
       7,   445,   350,   110,   312,   356,   372,   373,   374,   441,
     416,   351,    10,   352,   102,   362,   457,   363,   462,   364,
     365,   452,   375,   366,   107,   386,   387,   390,   464,   395,
     406,   407,   411,   414,   431,   421,   291,   415,   419,   420,
     422,   434,   432,   447,   433,   453,   454,   399,    52,   402,
     403,   461,   446,    91,    92,   330,   333,    93,   426,   428,
     331,    94,   332,   375,   424,   425,    95,   458,   334,   274,
     310,   335,   338,   209,   240,   435,   436,   317,   337,   412,
      37,   308,   405,   410,    51,    34,     0,    59,   429,   430,
     423,   443,   444,     0,   455,   456,   210,   211,     0,   212,
       0,   213,   214,   215,   216,   217,     6,     0,   151,   152,
     153,   218,   219,     0,     0,     0,   154,     0,     0,    10,
     155,   156,   157,   158,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   159,     0,   160,     0,
       0,   161,     0,     0,    48,     0,    49,    50,   162,   151,
     152,   153,     0,     0,     0,     0,     0,   154,   121,     0,
      10,   155,   156,   157,   158,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   159,     0,   160,
       0,     0,   161,     0,     0,    48,     0,    49,    50,   162,
     151,   152,   153,     0,     0,   193,     0,     0,   154,     0,
       0,    10,   155,   156,   157,   158,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   159,   190,
     160,     0,     0,   161,     0,     0,    48,     0,    49,    50,
     162,   151,   152,   153,     0,     0,     0,     0,     0,   154,
       0,     0,    10,   155,   156,   157,   158,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   159,
       0,   160,     0,     0,   161,     0,     0,    48,     0,    49,
      50,   162,  -236,     0,     0,     1,     0,     2,     3,     4,
       5,     0,     0,     0,     0,    79,    80,    81,    82,    83,
      84,     6,     7,     0,     0,     0,     0,     0,     8,     0,
       9,     0,    85,    86,    10,    87,    88,    89,    90
};

static const yytype_int16 yycheck[] =
{
       0,     1,     2,     3,     4,    31,    59,    35,   111,   121,
     354,   121,    30,   299,   113,   244,   162,   246,    47,   159,
     160,    10,   139,    23,    41,   342,    43,   208,    54,   101,
      32,    31,    30,    49,   180,    35,    36,    32,   436,    39,
      32,    41,    32,    43,    44,    63,   106,    49,   108,    32,
      78,   253,    54,    69,    54,    57,    56,   259,   456,    49,
     262,   121,    57,    65,    59,    60,   383,    57,     4,   141,
     356,    97,    49,    65,   102,    75,    49,    49,    78,   178,
      32,    32,    71,    64,   428,    66,   315,   116,   105,   117,
      66,    68,    49,   322,   111,    68,    68,    97,    50,    50,
       0,   101,   102,   131,   214,   105,   134,    49,   289,    68,
      37,   111,    71,    39,    40,    41,    43,   117,    64,   119,
      68,   121,   268,    71,   236,    49,   236,   127,    31,   358,
     247,   131,   161,   250,   134,    66,   276,    62,    63,   256,
       0,   141,    68,     3,    70,     5,     6,     7,     8,    44,
      45,   211,    49,   213,    64,   215,   216,   217,    64,    19,
      20,    66,    68,    69,    56,    37,    26,    32,    28,    64,
      65,    64,    32,   354,    59,    60,   236,    49,    55,    51,
     290,    53,    50,   293,    49,    57,    58,    68,   205,    54,
      46,    47,    57,    37,    38,    49,    68,    51,   427,    53,
     260,    37,    56,    57,    58,   205,     6,     7,   437,   312,
     210,    68,   212,    49,    65,    51,    55,    53,    38,   279,
      38,    57,    58,    50,   252,    50,    73,    68,    68,    37,
     459,   460,    68,    50,    55,    50,   236,    67,   467,   468,
      50,   340,   242,    51,    55,    53,   306,   428,    64,    57,
      58,    65,   252,   253,    51,   354,    53,    56,    55,   259,
      57,    58,   262,    56,    42,   318,    55,    51,   285,    53,
      37,    50,   300,    57,    58,    55,    52,   387,    55,    52,
      68,    64,    49,    57,    51,   285,    53,   347,    12,   349,
      57,    58,    24,   392,    50,   312,    71,    52,    37,   299,
     300,    68,   362,   363,    72,   365,     6,     7,     8,    51,
      53,    53,   312,    14,    13,    57,    58,    51,   435,    53,
      20,   431,    68,    57,    58,    12,    26,    27,    28,   428,
     390,    68,    32,    68,    64,    49,   446,    55,   455,    52,
      48,   440,   342,    68,    49,    68,    11,    13,   458,    68,
      27,    66,    72,    48,   414,    13,   356,    68,    68,    68,
      50,   421,    48,    48,    68,    68,    68,   363,    31,   369,
     370,    68,   432,    53,    53,   269,   272,    53,   404,   407,
     270,    53,   271,   383,   402,   403,    53,   447,   274,   187,
     242,   275,   279,   119,   127,   424,   425,   252,   278,   387,
      23,   236,   371,   383,   404,     7,    -1,   407,   408,   409,
     401,   429,   430,    -1,   443,   444,     9,    10,    -1,    12,
      -1,    14,    15,    16,    17,    18,    19,    -1,    21,    22,
      23,    24,    25,    -1,    -1,    -1,    29,    -1,    -1,    32,
      33,    34,    35,    36,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    49,    -1,    51,    -1,
      -1,    54,    -1,    -1,    57,    -1,    59,    60,    61,    21,
      22,    23,    -1,    -1,    -1,    -1,    -1,    29,    71,    -1,
      32,    33,    34,    35,    36,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    49,    -1,    51,
      -1,    -1,    54,    -1,    -1,    57,    -1,    59,    60,    61,
      21,    22,    23,    -1,    -1,    67,    -1,    -1,    29,    -1,
      -1,    32,    33,    34,    35,    36,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    49,    50,
      51,    -1,    -1,    54,    -1,    -1,    57,    -1,    59,    60,
      61,    21,    22,    23,    -1,    -1,    -1,    -1,    -1,    29,
      -1,    -1,    32,    33,    34,    35,    36,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    49,
      -1,    51,    -1,    -1,    54,    -1,    -1,    57,    -1,    59,
      60,    61,     0,    -1,    -1,     3,    -1,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    42,    43,    44,    45,    46,
      47,    19,    20,    -1,    -1,    -1,    -1,    -1,    26,    -1,
      28,    -1,    59,    60,    32,    62,    63,    64,    65
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     5,     6,     7,     8,    19,    20,    26,    28,
      32,    75,   135,   138,   140,   149,   153,   155,   161,   162,
     166,   167,   168,   189,   190,   191,   192,    75,    75,    75,
      75,    49,    30,   203,   203,    66,     4,   190,     0,    49,
     121,    64,    68,    69,    64,   127,   130,   130,    57,    59,
      60,    75,    95,   120,   152,    31,    49,    54,    57,    75,
     204,   205,   206,    75,    50,    75,   122,   123,   150,   204,
     204,    65,    75,   128,   129,    49,   117,   117,    66,    42,
      43,    44,    45,    46,    47,    59,    60,    62,    63,    64,
      65,    97,    99,   101,   103,   105,   107,   151,   120,   206,
     207,    49,    64,   131,   134,    56,    37,    49,    51,    53,
      57,    58,    68,   116,   124,   197,   130,    66,    50,    55,
      68,    71,   199,   200,    68,    68,    65,    55,    50,   119,
     120,    38,   193,   194,    38,   195,   196,   206,   120,    50,
      51,    55,    50,   207,   208,    65,   132,   133,   206,    73,
     204,    21,    22,    23,    29,    33,    34,    35,    36,    49,
      51,    54,    61,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    93,    94,
      95,    96,    98,   100,   102,   104,   106,   108,   110,   111,
      50,   111,   113,    67,   111,   114,   115,   126,    57,   204,
     205,    68,   124,    68,    68,    69,   117,   118,   206,   122,
       9,    10,    12,    14,    15,    16,    17,    18,    24,    25,
      75,   111,   137,   167,   170,   171,   172,   174,   175,   176,
     177,   178,   179,   180,   181,   182,   183,   184,   185,   199,
     128,    50,    55,   206,   139,   206,   141,    50,   194,   207,
      50,    65,    55,    56,   113,   113,   117,    96,   134,    56,
      51,    53,    56,    57,    58,    92,   124,    96,    97,    99,
     101,   103,   105,    37,   107,   109,    55,    50,    55,    55,
      52,    52,    57,    68,   204,    64,   164,   197,   198,    75,
     111,    75,   173,   111,   199,   111,   111,   112,   111,    12,
      66,    39,    40,    41,    68,    70,   169,    24,   184,    72,
     119,   200,    58,   200,   194,   156,   194,   132,    75,    81,
      50,    52,   194,    53,    81,    90,    91,   111,    81,    96,
      98,   100,   102,   104,   106,   110,   113,   126,   115,   165,
     204,    68,    71,   201,   202,   197,   199,    13,   199,    14,
      68,    68,    68,   173,   206,   111,    12,   205,   154,   200,
     134,   200,    49,    55,    52,    48,    68,   124,   125,     6,
       7,     8,    26,    27,    28,    75,   136,   142,   143,   146,
     157,   159,   168,   186,   187,   188,    68,    11,   111,   111,
      13,    68,   116,   124,   197,    68,   173,   200,   111,    91,
     111,   163,    75,    75,    37,   193,    27,    66,     6,     7,
     187,    72,   172,   199,    48,    68,   111,    68,   124,    68,
      68,    13,    50,   202,   130,   130,   120,   160,   206,    75,
      75,   111,    48,    68,   111,   117,   117,   158,   200,    68,
     116,   124,   197,   130,   130,   199,   111,    48,   194,   196,
     200,    68,   124,    68,    68,   117,   117,   199,   111,   145,
     148,    68,   194,   196,   199,   200,   200,   144,   147,   200,
     200
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    74,    75,    76,    77,    78,    79,    80,    80,    80,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      89,    89,    89,    89,    89,    89,    89,    90,    90,    91,
      91,    92,    93,    93,    93,    93,    93,    93,    93,    93,
      94,    94,    95,    95,    95,    96,    96,    96,    97,    97,
      98,    98,    99,    99,   100,   100,   101,   101,   101,   101,
     102,   102,   103,   103,   104,   104,   105,   106,   106,   107,
     108,   108,   109,   110,   110,   111,   112,   112,   113,   113,
     114,   115,   115,   116,   117,   117,   118,   118,   119,   119,
     120,   121,   121,   122,   122,   123,   124,   124,   125,   125,
     126,   126,   127,   127,   128,   128,   129,   130,   130,   131,
     131,   132,   132,   133,   134,   134,   135,   135,   135,   135,
     135,   136,   136,   136,   136,   136,   137,   137,   137,   137,
     137,   137,   139,   138,   141,   140,   142,   142,   142,   142,
     144,   143,   145,   143,   147,   146,   148,   146,   150,   149,
     151,   151,   151,   151,   151,   151,   152,   154,   153,   156,
     155,   158,   157,   160,   159,   161,   161,   161,   163,   162,
     164,   162,   165,   162,   166,   166,   166,   167,   168,   169,
     169,   169,   169,   170,   171,   172,   172,   172,   173,   174,
     175,   176,   176,   177,   178,   179,   180,   181,   182,   183,
     183,   183,   183,   183,   183,   183,   183,   183,   183,   183,
     183,   183,   184,   184,   185,   185,   186,   186,   186,   186,
     186,   186,   187,   187,   188,   188,   189,   189,   189,   189,
     189,   189,   189,   189,   189,   189,   190,   190,   191,   191,
     192,   193,   194,   194,   195,   195,   195,   196,   196,   197,
     198,   198,   199,   200,   200,   201,   202,   202,   203,   204,
     204,   205,   205,   206,   206,   206,   206,   206,   206,   206,
     206,   206,   207,   207,   208,   208
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     2,     3,     3,     6,     4,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     3,     1,     1,
       3,     3,     1,     3,     3,     3,     2,     2,     2,     2,
       1,     2,     1,     1,     1,     1,     2,     2,     1,     1,
       1,     3,     1,     1,     1,     3,     1,     1,     1,     1,
       1,     3,     1,     1,     1,     3,     1,     1,     3,     1,
       1,     3,     1,     1,     3,     1,     1,     0,     1,     3,
       1,     1,     3,     3,     2,     3,     1,     0,     1,     3,
       3,     2,     3,     1,     3,     4,     2,     3,     1,     0,
       1,     3,     2,     3,     1,     3,     1,     1,     0,     2,
       3,     1,     3,     1,     1,     0,     4,     5,     5,     5,
       6,     4,     5,     5,     5,     6,     4,     4,     5,     5,
       5,     6,     0,     7,     0,     7,     1,     1,     1,     2,
       0,     8,     0,     7,     0,     8,     0,     7,     0,     5,
       1,     1,     1,     1,     1,     1,     1,     0,     9,     0,
       8,     0,     5,     0,     4,     1,     1,     0,     0,    10,
       0,     7,     0,     8,     5,     5,     3,     2,     2,     1,
       1,     1,     1,     4,     2,     5,     5,     3,     1,     7,
       1,     9,     8,     3,     5,     1,     3,     3,     3,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     2,     1,     0,     1,     1,     1,     1,
       1,     1,     1,     2,     1,     0,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     2,     1,     0,
       1,     2,     1,     0,     2,     4,     3,     1,     0,     2,
       1,     0,     3,     1,     1,     3,     1,     1,     2,     2,
       3,     1,     3,     1,     2,     4,     2,     3,     1,     3,
       4,     5,     1,     3,     1,     0
};


/* YYDPREC[RULE-NUM] -- Dynamic precedence of rule #RULE-NUM (0 if none).  */
static const yytype_int8 yydprec[] =
{
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       3,     2,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0
};

/* YYMERGER[RULE-NUM] -- Index of merging function for rule #RULE-NUM.  */
static const yytype_int8 yymerger[] =
{
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0
};

/* YYIMMEDIATE[RULE-NUM] -- True iff rule #RULE-NUM is not to be deferred, as
   in the case of predicates.  */
static const yybool yyimmediate[] =
{
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0
};

/* YYCONFLP[YYPACT[STATE-NUM]] -- Pointer into YYCONFL of start of
   list of conflicting reductions corresponding to action entry for
   state STATE-NUM in yytable.  0 means no conflicts.  The list in
   yyconfl is terminated by a rule number of 0.  */
static const yytype_int8 yyconflp[] =
{
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     7,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     1,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     3,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     5,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     9,
      17,    11,     0,     0,     0,    13,    15,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    19,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0
};

/* YYCONFL[I] -- lists of conflicting rule numbers, each terminated by
   0, pointed into by YYCONFLP.  */
static const short yyconfl[] =
{
       0,   115,     0,   108,     0,   115,     0,   115,     0,   241,
       0,   241,     0,   241,     0,   241,     0,    11,     0,   115,
       0
};

/* Error token number */
#define YYTERROR 1


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)                                \
    do                                                                  \
      if (N)                                                            \
        {                                                               \
          (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;        \
          (Current).first_column = YYRHSLOC (Rhs, 1).first_column;      \
          (Current).last_line    = YYRHSLOC (Rhs, N).last_line;         \
          (Current).last_column  = YYRHSLOC (Rhs, N).last_column;       \
        }                                                               \
      else                                                              \
        {                                                               \
          (Current).first_line   = (Current).last_line   =              \
            YYRHSLOC (Rhs, 0).last_line;                                \
          (Current).first_column = (Current).last_column =              \
            YYRHSLOC (Rhs, 0).last_column;                              \
        }                                                               \
    while (0)
#endif

# define YYRHSLOC(Rhs, K) ((Rhs)[K].yystate.yyloc)


YYSTYPE yylval;
YYLTYPE yylloc;

int yynerrs;
int yychar;

static const int YYEOF = 0;
static const int YYEMPTY = -2;

typedef enum { yyok, yyaccept, yyabort, yyerr } YYRESULTTAG;

#define YYCHK(YYE)                              \
  do {                                          \
    YYRESULTTAG yychk_flag = YYE;               \
    if (yychk_flag != yyok)                     \
      return yychk_flag;                        \
  } while (0)

#if YYDEBUG

# ifndef YYFPRINTF
#  define YYFPRINTF fprintf
# endif

# define YY_FPRINTF                             \
  YY_IGNORE_USELESS_CAST_BEGIN YY_FPRINTF_

# define YY_FPRINTF_(Args)                      \
  do {                                          \
    YYFPRINTF Args;                             \
    YY_IGNORE_USELESS_CAST_END                  \
  } while (0)

# define YY_DPRINTF                             \
  YY_IGNORE_USELESS_CAST_BEGIN YY_DPRINTF_

# define YY_DPRINTF_(Args)                      \
  do {                                          \
    if (yydebug)                                \
      YYFPRINTF Args;                           \
    YY_IGNORE_USELESS_CAST_END                  \
  } while (0)


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL

/* Print *YYLOCP on YYO.  Private, do not rely on its existence. */

YY_ATTRIBUTE_UNUSED
static int
yy_location_print_ (FILE *yyo, YYLTYPE const * const yylocp)
{
  int res = 0;
  int end_col = 0 != yylocp->last_column ? yylocp->last_column - 1 : 0;
  if (0 <= yylocp->first_line)
    {
      res += YYFPRINTF (yyo, "%d", yylocp->first_line);
      if (0 <= yylocp->first_column)
        res += YYFPRINTF (yyo, ".%d", yylocp->first_column);
    }
  if (0 <= yylocp->last_line)
    {
      if (yylocp->first_line < yylocp->last_line)
        {
          res += YYFPRINTF (yyo, "-%d", yylocp->last_line);
          if (0 <= end_col)
            res += YYFPRINTF (yyo, ".%d", end_col);
        }
      else if (0 <= end_col && yylocp->first_column < end_col)
        res += YYFPRINTF (yyo, "-%d", end_col);
    }
  return res;
 }

#  define YY_LOCATION_PRINT(File, Loc)          \
  yy_location_print_ (File, &(Loc))

# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif



/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
{
  FILE *yyoutput = yyo;
  YYUSE (yyoutput);
  YYUSE (yylocationp);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
{
  YYFPRINTF (yyo, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  YY_LOCATION_PRINT (yyo, *yylocationp);
  YYFPRINTF (yyo, ": ");
  yy_symbol_value_print (yyo, yytype, yyvaluep, yylocationp);
  YYFPRINTF (yyo, ")");
}

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                  \
  do {                                                                  \
    if (yydebug)                                                        \
      {                                                                 \
        YY_FPRINTF ((stderr, "%s ", Title));                            \
        yy_symbol_print (stderr, Type, Value, Location);        \
        YY_FPRINTF ((stderr, "\n"));                                    \
      }                                                                 \
  } while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;

struct yyGLRStack;
static void yypstack (struct yyGLRStack* yystackp, ptrdiff_t yyk)
  YY_ATTRIBUTE_UNUSED;
static void yypdumpstack (struct yyGLRStack* yystackp)
  YY_ATTRIBUTE_UNUSED;

#else /* !YYDEBUG */

# define YY_DPRINTF(Args) do {} while (yyfalse)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)

#endif /* !YYDEBUG */

/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   SIZE_MAX < YYMAXDEPTH * sizeof (GLRStackItem)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif

/* Minimum number of free items on the stack allowed after an
   allocation.  This is to allow allocation and initialization
   to be completed by functions that call yyexpandGLRStack before the
   stack is expanded, thus insuring that all necessary pointers get
   properly redirected to new data.  */
#define YYHEADROOM 2

#ifndef YYSTACKEXPANDABLE
#  define YYSTACKEXPANDABLE 1
#endif

#if YYSTACKEXPANDABLE
# define YY_RESERVE_GLRSTACK(Yystack)                   \
  do {                                                  \
    if (Yystack->yyspaceLeft < YYHEADROOM)              \
      yyexpandGLRStack (Yystack);                       \
  } while (0)
#else
# define YY_RESERVE_GLRSTACK(Yystack)                   \
  do {                                                  \
    if (Yystack->yyspaceLeft < YYHEADROOM)              \
      yyMemoryExhausted (Yystack);                      \
  } while (0)
#endif


#if YYERROR_VERBOSE

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static ptrdiff_t
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      ptrdiff_t yyn = 0;
      char const *yyp = yystr;

      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            else
              goto append;

          append:
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (yyres)
    return yystpcpy (yyres, yystr) - yyres;
  else
    return YY_CAST (ptrdiff_t, strlen (yystr));
}
# endif

#endif /* !YYERROR_VERBOSE */

/** State numbers. */
typedef int yyStateNum;

/** Rule numbers. */
typedef int yyRuleNum;

/** Grammar symbol. */
typedef int yySymbol;

/** Item references. */
typedef short yyItemNum;

typedef struct yyGLRState yyGLRState;
typedef struct yyGLRStateSet yyGLRStateSet;
typedef struct yySemanticOption yySemanticOption;
typedef union yyGLRStackItem yyGLRStackItem;
typedef struct yyGLRStack yyGLRStack;

struct yyGLRState {
  /** Type tag: always true.  */
  yybool yyisState;
  /** Type tag for yysemantics.  If true, yysval applies, otherwise
   *  yyfirstVal applies.  */
  yybool yyresolved;
  /** Number of corresponding LALR(1) machine state.  */
  yyStateNum yylrState;
  /** Preceding state in this stack */
  yyGLRState* yypred;
  /** Source position of the last token produced by my symbol */
  ptrdiff_t yyposn;
  union {
    /** First in a chain of alternative reductions producing the
     *  nonterminal corresponding to this state, threaded through
     *  yynext.  */
    yySemanticOption* yyfirstVal;
    /** Semantic value for this state.  */
    YYSTYPE yysval;
  } yysemantics;
  /** Source location for this state.  */
  YYLTYPE yyloc;
};

struct yyGLRStateSet {
  yyGLRState** yystates;
  /** During nondeterministic operation, yylookaheadNeeds tracks which
   *  stacks have actually needed the current lookahead.  During deterministic
   *  operation, yylookaheadNeeds[0] is not maintained since it would merely
   *  duplicate yychar != YYEMPTY.  */
  yybool* yylookaheadNeeds;
  ptrdiff_t yysize;
  ptrdiff_t yycapacity;
};

struct yySemanticOption {
  /** Type tag: always false.  */
  yybool yyisState;
  /** Rule number for this reduction */
  yyRuleNum yyrule;
  /** The last RHS state in the list of states to be reduced.  */
  yyGLRState* yystate;
  /** The lookahead for this reduction.  */
  int yyrawchar;
  YYSTYPE yyval;
  YYLTYPE yyloc;
  /** Next sibling in chain of options.  To facilitate merging,
   *  options are chained in decreasing order by address.  */
  yySemanticOption* yynext;
};

/** Type of the items in the GLR stack.  The yyisState field
 *  indicates which item of the union is valid.  */
union yyGLRStackItem {
  yyGLRState yystate;
  yySemanticOption yyoption;
};

struct yyGLRStack {
  int yyerrState;
  /* To compute the location of the error token.  */
  yyGLRStackItem yyerror_range[3];

  YYJMP_BUF yyexception_buffer;
  yyGLRStackItem* yyitems;
  yyGLRStackItem* yynextFree;
  ptrdiff_t yyspaceLeft;
  yyGLRState* yysplitPoint;
  yyGLRState* yylastDeleted;
  yyGLRStateSet yytops;
};

#if YYSTACKEXPANDABLE
static void yyexpandGLRStack (yyGLRStack* yystackp);
#endif

_Noreturn static void
yyFail (yyGLRStack* yystackp, const char* yymsg)
{
  if (yymsg != YY_NULLPTR)
    yyerror (yymsg);
  YYLONGJMP (yystackp->yyexception_buffer, 1);
}

_Noreturn static void
yyMemoryExhausted (yyGLRStack* yystackp)
{
  YYLONGJMP (yystackp->yyexception_buffer, 2);
}

#if YYDEBUG || YYERROR_VERBOSE
/** A printable representation of TOKEN.  */
static inline const char*
yytokenName (yySymbol yytoken)
{
  return yytoken == YYEMPTY ? "" : yytname[yytoken];
}
#endif

/** Fill in YYVSP[YYLOW1 .. YYLOW0-1] from the chain of states starting
 *  at YYVSP[YYLOW0].yystate.yypred.  Leaves YYVSP[YYLOW1].yystate.yypred
 *  containing the pointer to the next state in the chain.  */
static void yyfillin (yyGLRStackItem *, int, int) YY_ATTRIBUTE_UNUSED;
static void
yyfillin (yyGLRStackItem *yyvsp, int yylow0, int yylow1)
{
  int i;
  yyGLRState *s = yyvsp[yylow0].yystate.yypred;
  for (i = yylow0-1; i >= yylow1; i -= 1)
    {
#if YYDEBUG
      yyvsp[i].yystate.yylrState = s->yylrState;
#endif
      yyvsp[i].yystate.yyresolved = s->yyresolved;
      if (s->yyresolved)
        yyvsp[i].yystate.yysemantics.yysval = s->yysemantics.yysval;
      else
        /* The effect of using yysval or yyloc (in an immediate rule) is
         * undefined.  */
        yyvsp[i].yystate.yysemantics.yyfirstVal = YY_NULLPTR;
      yyvsp[i].yystate.yyloc = s->yyloc;
      s = yyvsp[i].yystate.yypred = s->yypred;
    }
}


/** If yychar is empty, fetch the next token.  */
static inline yySymbol
yygetToken (int *yycharp)
{
  yySymbol yytoken;
  if (*yycharp == YYEMPTY)
    {
      YY_DPRINTF ((stderr, "Reading a token: "));
      *yycharp = yylex ();
    }
  if (*yycharp <= YYEOF)
    {
      *yycharp = yytoken = YYEOF;
      YY_DPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (*yycharp);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }
  return yytoken;
}

/* Do nothing if YYNORMAL or if *YYLOW <= YYLOW1.  Otherwise, fill in
 * YYVSP[YYLOW1 .. *YYLOW-1] as in yyfillin and set *YYLOW = YYLOW1.
 * For convenience, always return YYLOW1.  */
static inline int yyfill (yyGLRStackItem *, int *, int, yybool)
     YY_ATTRIBUTE_UNUSED;
static inline int
yyfill (yyGLRStackItem *yyvsp, int *yylow, int yylow1, yybool yynormal)
{
  if (!yynormal && yylow1 < *yylow)
    {
      yyfillin (yyvsp, *yylow, yylow1);
      *yylow = yylow1;
    }
  return yylow1;
}

/** Perform user action for rule number YYN, with RHS length YYRHSLEN,
 *  and top stack item YYVSP.  YYLVALP points to place to put semantic
 *  value ($$), and yylocp points to place for location information
 *  (@$).  Returns yyok for normal return, yyaccept for YYACCEPT,
 *  yyerr for YYERROR, yyabort for YYABORT.  */
static YYRESULTTAG
yyuserAction (yyRuleNum yyn, int yyrhslen, yyGLRStackItem* yyvsp,
              yyGLRStack* yystackp,
              YYSTYPE* yyvalp, YYLTYPE *yylocp)
{
  yybool yynormal YY_ATTRIBUTE_UNUSED = yystackp->yysplitPoint == YY_NULLPTR;
  int yylow;
  YYUSE (yyvalp);
  YYUSE (yylocp);
  YYUSE (yyrhslen);
# undef yyerrok
# define yyerrok (yystackp->yyerrState = 0)
# undef YYACCEPT
# define YYACCEPT return yyaccept
# undef YYABORT
# define YYABORT return yyabort
# undef YYERROR
# define YYERROR return yyerrok, yyerr
# undef YYRECOVERING
# define YYRECOVERING() (yystackp->yyerrState != 0)
# undef yyclearin
# define yyclearin (yychar = YYEMPTY)
# undef YYFILL
# define YYFILL(N) yyfill (yyvsp, &yylow, (N), yynormal)
# undef YYBACKUP
# define YYBACKUP(Token, Value)                                              \
  return yyerror (YY_("syntax error: cannot back up")),     \
         yyerrok, yyerr

  yylow = 1;
  if (yyrhslen == 0)
    *yyvalp = yyval_default;
  else
    *yyvalp = yyvsp[YYFILL (1-yyrhslen)].yystate.yysemantics.yysval;
  /* Default location. */
  YYLLOC_DEFAULT ((*yylocp), (yyvsp - yyrhslen), yyrhslen);
  yystackp->yyerror_range[1].yystate.yyloc = *yylocp;

  switch (yyn)
    {
  case 2:
#line 159 "bi/parser.ypp"
            { ((*yyvalp).valName) = new bi::Name((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valString)); }
#line 1628 "bi/parser.cpp"
    break;

  case 3:
#line 168 "bi/parser.ypp"
                    { ((*yyvalp).valExpression) = new bi::Literal<bool>((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valString), make_loc((*yylocp))); }
#line 1634 "bi/parser.cpp"
    break;

  case 4:
#line 172 "bi/parser.ypp"
                   { ((*yyvalp).valExpression) = new bi::Literal<int64_t>((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valString), make_loc((*yylocp))); }
#line 1640 "bi/parser.cpp"
    break;

  case 5:
#line 176 "bi/parser.ypp"
                    { ((*yyvalp).valExpression) = new bi::Literal<double>((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valString), make_loc((*yylocp))); }
#line 1646 "bi/parser.cpp"
    break;

  case 6:
#line 180 "bi/parser.ypp"
                      { ((*yyvalp).valExpression) = new bi::Literal<const char*>((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valString), make_loc((*yylocp))); }
#line 1652 "bi/parser.cpp"
    break;

  case 11:
#line 191 "bi/parser.ypp"
                                       { ((*yyvalp).valExpression) = new bi::NamedExpression((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 1658 "bi/parser.cpp"
    break;

  case 12:
#line 195 "bi/parser.ypp"
                               { ((*yyvalp).valExpression) = new bi::Parentheses((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1664 "bi/parser.cpp"
    break;

  case 13:
#line 199 "bi/parser.ypp"
                               { ((*yyvalp).valExpression) = new bi::Sequence((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1670 "bi/parser.cpp"
    break;

  case 14:
#line 203 "bi/parser.ypp"
                                                              { ((*yyvalp).valExpression) = new bi::Cast(new bi::NamedType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1676 "bi/parser.cpp"
    break;

  case 15:
#line 207 "bi/parser.ypp"
                                                            { ((*yyvalp).valExpression) = new bi::LambdaFunction((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 1682 "bi/parser.cpp"
    break;

  case 16:
#line 211 "bi/parser.ypp"
            { ((*yyvalp).valExpression) = new bi::This(make_loc((*yylocp))); }
#line 1688 "bi/parser.cpp"
    break;

  case 17:
#line 215 "bi/parser.ypp"
             { ((*yyvalp).valExpression) = new bi::Super(make_loc((*yylocp))); }
#line 1694 "bi/parser.cpp"
    break;

  case 18:
#line 219 "bi/parser.ypp"
           { ((*yyvalp).valExpression) = new bi::Nil(make_loc((*yylocp))); }
#line 1700 "bi/parser.cpp"
    break;

  case 27:
#line 234 "bi/parser.ypp"
                                      { ((*yyvalp).valExpression) = new bi::Range((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1706 "bi/parser.cpp"
    break;

  case 28:
#line 235 "bi/parser.ypp"
                                      { ((*yyvalp).valExpression) = new bi::Index((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1712 "bi/parser.cpp"
    break;

  case 30:
#line 240 "bi/parser.ypp"
                                       { ((*yyvalp).valExpression) = new bi::ExpressionList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1718 "bi/parser.cpp"
    break;

  case 31:
#line 244 "bi/parser.ypp"
                          { ((*yyvalp).valExpression) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1724 "bi/parser.cpp"
    break;

  case 33:
#line 249 "bi/parser.ypp"
                                         { ((*yyvalp).valExpression) = new bi::Member((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1730 "bi/parser.cpp"
    break;

  case 34:
#line 250 "bi/parser.ypp"
                                         { ((*yyvalp).valExpression) = new bi::Global((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1736 "bi/parser.cpp"
    break;

  case 35:
#line 251 "bi/parser.ypp"
                                         { ((*yyvalp).valExpression) = new bi::Member((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1742 "bi/parser.cpp"
    break;

  case 36:
#line 252 "bi/parser.ypp"
                                         { ((*yyvalp).valExpression) = new bi::Slice((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1748 "bi/parser.cpp"
    break;

  case 37:
#line 253 "bi/parser.ypp"
                                         { ((*yyvalp).valExpression) = new bi::Call((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1754 "bi/parser.cpp"
    break;

  case 38:
#line 254 "bi/parser.ypp"
                                         { ((*yyvalp).valExpression) = new bi::Get((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1760 "bi/parser.cpp"
    break;

  case 39:
#line 255 "bi/parser.ypp"
                                         { ((*yyvalp).valExpression) = new bi::GetReturn((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1766 "bi/parser.cpp"
    break;

  case 41:
#line 262 "bi/parser.ypp"
                              { ((*yyvalp).valExpression) = new bi::Query((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1772 "bi/parser.cpp"
    break;

  case 42:
#line 266 "bi/parser.ypp"
           { ((*yyvalp).valName) = new bi::Name("+"); }
#line 1778 "bi/parser.cpp"
    break;

  case 43:
#line 267 "bi/parser.ypp"
           { ((*yyvalp).valName) = new bi::Name("-"); }
#line 1784 "bi/parser.cpp"
    break;

  case 44:
#line 268 "bi/parser.ypp"
           { ((*yyvalp).valName) = new bi::Name("!"); }
#line 1790 "bi/parser.cpp"
    break;

  case 46:
#line 273 "bi/parser.ypp"
                                         { ((*yyvalp).valExpression) = new bi::UnaryCall((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1796 "bi/parser.cpp"
    break;

  case 47:
#line 274 "bi/parser.ypp"
                                         { ((*yyvalp).valExpression) = new bi::Spin((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1802 "bi/parser.cpp"
    break;

  case 48:
#line 278 "bi/parser.ypp"
           { ((*yyvalp).valName) = new bi::Name("*"); }
#line 1808 "bi/parser.cpp"
    break;

  case 49:
#line 279 "bi/parser.ypp"
           { ((*yyvalp).valName) = new bi::Name("/"); }
#line 1814 "bi/parser.cpp"
    break;

  case 51:
#line 284 "bi/parser.ypp"
                                                                           { ((*yyvalp).valExpression) = new bi::BinaryCall((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1820 "bi/parser.cpp"
    break;

  case 52:
#line 288 "bi/parser.ypp"
           { ((*yyvalp).valName) = new bi::Name("+"); }
#line 1826 "bi/parser.cpp"
    break;

  case 53:
#line 289 "bi/parser.ypp"
           { ((*yyvalp).valName) = new bi::Name("-"); }
#line 1832 "bi/parser.cpp"
    break;

  case 55:
#line 294 "bi/parser.ypp"
                                                                       { ((*yyvalp).valExpression) = new bi::BinaryCall((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1838 "bi/parser.cpp"
    break;

  case 56:
#line 298 "bi/parser.ypp"
             { ((*yyvalp).valName) = new bi::Name("<"); }
#line 1844 "bi/parser.cpp"
    break;

  case 57:
#line 299 "bi/parser.ypp"
             { ((*yyvalp).valName) = new bi::Name(">"); }
#line 1850 "bi/parser.cpp"
    break;

  case 58:
#line 300 "bi/parser.ypp"
             { ((*yyvalp).valName) = new bi::Name("<="); }
#line 1856 "bi/parser.cpp"
    break;

  case 59:
#line 301 "bi/parser.ypp"
             { ((*yyvalp).valName) = new bi::Name(">="); }
#line 1862 "bi/parser.cpp"
    break;

  case 61:
#line 311 "bi/parser.ypp"
                                                                               { ((*yyvalp).valExpression) = new bi::BinaryCall((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1868 "bi/parser.cpp"
    break;

  case 62:
#line 315 "bi/parser.ypp"
             { ((*yyvalp).valName) = new bi::Name("=="); }
#line 1874 "bi/parser.cpp"
    break;

  case 63:
#line 316 "bi/parser.ypp"
             { ((*yyvalp).valName) = new bi::Name("!="); }
#line 1880 "bi/parser.cpp"
    break;

  case 65:
#line 321 "bi/parser.ypp"
                                                                   { ((*yyvalp).valExpression) = new bi::BinaryCall((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1886 "bi/parser.cpp"
    break;

  case 66:
#line 325 "bi/parser.ypp"
              { ((*yyvalp).valName) = new bi::Name("&&"); }
#line 1892 "bi/parser.cpp"
    break;

  case 68:
#line 330 "bi/parser.ypp"
                                                                       { ((*yyvalp).valExpression) = new bi::BinaryCall((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1898 "bi/parser.cpp"
    break;

  case 69:
#line 334 "bi/parser.ypp"
             { ((*yyvalp).valName) = new bi::Name("||"); }
#line 1904 "bi/parser.cpp"
    break;

  case 71:
#line 339 "bi/parser.ypp"
                                                                        { ((*yyvalp).valExpression) = new bi::BinaryCall((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1910 "bi/parser.cpp"
    break;

  case 72:
#line 343 "bi/parser.ypp"
                 { ((*yyvalp).valName) = new bi::Name("<-"); }
#line 1916 "bi/parser.cpp"
    break;

  case 74:
#line 348 "bi/parser.ypp"
                                                               { ((*yyvalp).valExpression) = new bi::Assign((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1922 "bi/parser.cpp"
    break;

  case 77:
#line 357 "bi/parser.ypp"
                  { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1928 "bi/parser.cpp"
    break;

  case 79:
#line 362 "bi/parser.ypp"
                                      { ((*yyvalp).valExpression) = new bi::ExpressionList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1934 "bi/parser.cpp"
    break;

  case 80:
#line 366 "bi/parser.ypp"
                   { ((*yyvalp).valExpression) = new bi::Span((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1940 "bi/parser.cpp"
    break;

  case 82:
#line 371 "bi/parser.ypp"
                                     { ((*yyvalp).valExpression) = new bi::ExpressionList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1946 "bi/parser.cpp"
    break;

  case 83:
#line 375 "bi/parser.ypp"
                         { ((*yyvalp).valExpression) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1952 "bi/parser.cpp"
    break;

  case 84:
#line 379 "bi/parser.ypp"
                              { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1958 "bi/parser.cpp"
    break;

  case 85:
#line 380 "bi/parser.ypp"
                              { ((*yyvalp).valExpression) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1964 "bi/parser.cpp"
    break;

  case 87:
#line 385 "bi/parser.ypp"
                  { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1970 "bi/parser.cpp"
    break;

  case 89:
#line 390 "bi/parser.ypp"
                                    { ((*yyvalp).valExpression) = new bi::ExpressionList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1976 "bi/parser.cpp"
    break;

  case 90:
#line 394 "bi/parser.ypp"
                     { ((*yyvalp).valExpression) = new bi::Parameter(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1982 "bi/parser.cpp"
    break;

  case 91:
#line 398 "bi/parser.ypp"
                           { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1988 "bi/parser.cpp"
    break;

  case 92:
#line 399 "bi/parser.ypp"
                           { ((*yyvalp).valExpression) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1994 "bi/parser.cpp"
    break;

  case 94:
#line 404 "bi/parser.ypp"
                              { ((*yyvalp).valExpression) = new bi::ExpressionList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2000 "bi/parser.cpp"
    break;

  case 95:
#line 408 "bi/parser.ypp"
                                    { ((*yyvalp).valExpression) = new bi::Parameter(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2006 "bi/parser.cpp"
    break;

  case 96:
#line 412 "bi/parser.ypp"
                               { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 2012 "bi/parser.cpp"
    break;

  case 97:
#line 413 "bi/parser.ypp"
                               { ((*yyvalp).valExpression) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 2018 "bi/parser.cpp"
    break;

  case 99:
#line 418 "bi/parser.ypp"
                 { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 2024 "bi/parser.cpp"
    break;

  case 100:
#line 422 "bi/parser.ypp"
                     { ((*yyvalp).valInt) = 1; }
#line 2030 "bi/parser.cpp"
    break;

  case 101:
#line 423 "bi/parser.ypp"
                     { ((*yyvalp).valInt) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valInt) + 1; }
#line 2036 "bi/parser.cpp"
    break;

  case 102:
#line 427 "bi/parser.ypp"
                            { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 2042 "bi/parser.cpp"
    break;

  case 103:
#line 428 "bi/parser.ypp"
                            { ((*yyvalp).valExpression) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 2048 "bi/parser.cpp"
    break;

  case 105:
#line 433 "bi/parser.ypp"
                                { ((*yyvalp).valExpression) = new bi::ExpressionList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2054 "bi/parser.cpp"
    break;

  case 106:
#line 437 "bi/parser.ypp"
            { ((*yyvalp).valExpression) = new bi::Generic(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valName), empty_type((*yylocp)), make_loc((*yylocp))); }
#line 2060 "bi/parser.cpp"
    break;

  case 108:
#line 442 "bi/parser.ypp"
                { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 2066 "bi/parser.cpp"
    break;

  case 109:
#line 446 "bi/parser.ypp"
                                     { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 2072 "bi/parser.cpp"
    break;

  case 110:
#line 447 "bi/parser.ypp"
                                     { ((*yyvalp).valType) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType); }
#line 2078 "bi/parser.cpp"
    break;

  case 112:
#line 452 "bi/parser.ypp"
                                                  { ((*yyvalp).valType) = new bi::TypeList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2084 "bi/parser.cpp"
    break;

  case 115:
#line 461 "bi/parser.ypp"
                         { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 2090 "bi/parser.cpp"
    break;

  case 116:
#line 470 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2096 "bi/parser.cpp"
    break;

  case 117:
#line 471 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2102 "bi/parser.cpp"
    break;

  case 118:
#line 472 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 2108 "bi/parser.cpp"
    break;

  case 119:
#line 473 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), new bi::ArrayType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression)->width(), make_loc((*yylocp))), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2114 "bi/parser.cpp"
    break;

  case 120:
#line 474 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), new bi::ArrayType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression)->width(), make_loc((*yylocp))), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2120 "bi/parser.cpp"
    break;

  case 121:
#line 478 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2126 "bi/parser.cpp"
    break;

  case 122:
#line 479 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2132 "bi/parser.cpp"
    break;

  case 123:
#line 480 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 2138 "bi/parser.cpp"
    break;

  case 124:
#line 481 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), new bi::ArrayType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression)->width(), make_loc((*yylocp))), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2144 "bi/parser.cpp"
    break;

  case 125:
#line 482 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), new bi::ArrayType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression)->width(), make_loc((*yylocp))), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2150 "bi/parser.cpp"
    break;

  case 126:
#line 486 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::LocalVariable(bi::AUTO, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), empty_type((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 2156 "bi/parser.cpp"
    break;

  case 127:
#line 487 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::LocalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2162 "bi/parser.cpp"
    break;

  case 128:
#line 488 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::LocalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2168 "bi/parser.cpp"
    break;

  case 129:
#line 489 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::LocalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 2174 "bi/parser.cpp"
    break;

  case 130:
#line 490 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::LocalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), new bi::ArrayType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression)->width(), make_loc((*yylocp))), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2180 "bi/parser.cpp"
    break;

  case 131:
#line 491 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::LocalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), new bi::ArrayType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression)->width(), make_loc((*yylocp))), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2186 "bi/parser.cpp"
    break;

  case 132:
#line 495 "bi/parser.ypp"
                                                                      { push_raw(); }
#line 2192 "bi/parser.cpp"
    break;

  case 133:
#line 495 "bi/parser.ypp"
                                                                                                       { ((*yyvalp).valStatement) = new bi::Function(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2198 "bi/parser.cpp"
    break;

  case 134:
#line 499 "bi/parser.ypp"
                                                                         { push_raw(); }
#line 2204 "bi/parser.cpp"
    break;

  case 135:
#line 499 "bi/parser.ypp"
                                                                                                          { ((*yyvalp).valStatement) = new bi::Fiber(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2210 "bi/parser.cpp"
    break;

  case 136:
#line 503 "bi/parser.ypp"
                      { ((*yyvalp).valAnnotation) = bi::ABSTRACT; }
#line 2216 "bi/parser.cpp"
    break;

  case 137:
#line 504 "bi/parser.ypp"
                      { ((*yyvalp).valAnnotation) = bi::FINAL; }
#line 2222 "bi/parser.cpp"
    break;

  case 138:
#line 505 "bi/parser.ypp"
                      { ((*yyvalp).valAnnotation) = bi::OVERRIDE; }
#line 2228 "bi/parser.cpp"
    break;

  case 139:
#line 506 "bi/parser.ypp"
                      { ((*yyvalp).valAnnotation) = bi::Annotation(bi::FINAL|bi::OVERRIDE); }
#line 2234 "bi/parser.cpp"
    break;

  case 140:
#line 512 "bi/parser.ypp"
                                                                                                 { push_raw(); }
#line 2240 "bi/parser.cpp"
    break;

  case 141:
#line 512 "bi/parser.ypp"
                                                                                                                                  { ((*yyvalp).valStatement) = new bi::MemberFunction((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-7)].yystate.yysemantics.yysval.valAnnotation), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2246 "bi/parser.cpp"
    break;

  case 142:
#line 513 "bi/parser.ypp"
                                                                      { push_raw(); }
#line 2252 "bi/parser.cpp"
    break;

  case 143:
#line 513 "bi/parser.ypp"
                                                                                                                                  { ((*yyvalp).valStatement) = new bi::MemberFunction(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2258 "bi/parser.cpp"
    break;

  case 144:
#line 517 "bi/parser.ypp"
                                                                                                    { push_raw(); }
#line 2264 "bi/parser.cpp"
    break;

  case 145:
#line 517 "bi/parser.ypp"
                                                                                                                                     { ((*yyvalp).valStatement) = new bi::MemberFiber((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-7)].yystate.yysemantics.yysval.valAnnotation), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2270 "bi/parser.cpp"
    break;

  case 146:
#line 518 "bi/parser.ypp"
                                                                         { push_raw(); }
#line 2276 "bi/parser.cpp"
    break;

  case 147:
#line 518 "bi/parser.ypp"
                                                                                                                                     { ((*yyvalp).valStatement) = new bi::MemberFiber(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2282 "bi/parser.cpp"
    break;

  case 148:
#line 522 "bi/parser.ypp"
                           { push_raw(); }
#line 2288 "bi/parser.cpp"
    break;

  case 149:
#line 522 "bi/parser.ypp"
                                                            { ((*yyvalp).valStatement) = new bi::Program((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2294 "bi/parser.cpp"
    break;

  case 157:
#line 539 "bi/parser.ypp"
                                                                                { push_raw(); }
#line 2300 "bi/parser.cpp"
    break;

  case 158:
#line 539 "bi/parser.ypp"
                                                                                                                 { ((*yyvalp).valStatement) = new bi::BinaryOperator(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2306 "bi/parser.cpp"
    break;

  case 159:
#line 543 "bi/parser.ypp"
                                                                     { push_raw(); }
#line 2312 "bi/parser.cpp"
    break;

  case 160:
#line 543 "bi/parser.ypp"
                                                                                                      { ((*yyvalp).valStatement) = new bi::UnaryOperator(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2318 "bi/parser.cpp"
    break;

  case 161:
#line 547 "bi/parser.ypp"
                                 { push_raw(); }
#line 2324 "bi/parser.cpp"
    break;

  case 162:
#line 547 "bi/parser.ypp"
                                                                  { ((*yyvalp).valStatement) = new bi::AssignmentOperator((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2330 "bi/parser.cpp"
    break;

  case 163:
#line 551 "bi/parser.ypp"
                           { push_raw(); }
#line 2336 "bi/parser.cpp"
    break;

  case 164:
#line 551 "bi/parser.ypp"
                                                            { ((*yyvalp).valStatement) = new bi::ConversionOperator((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2342 "bi/parser.cpp"
    break;

  case 165:
#line 555 "bi/parser.ypp"
                { ((*yyvalp).valAnnotation) = bi::FINAL; }
#line 2348 "bi/parser.cpp"
    break;

  case 166:
#line 556 "bi/parser.ypp"
                { ((*yyvalp).valAnnotation) = bi::ABSTRACT; }
#line 2354 "bi/parser.cpp"
    break;

  case 167:
#line 557 "bi/parser.ypp"
                { ((*yyvalp).valAnnotation) = bi::NONE; }
#line 2360 "bi/parser.cpp"
    break;

  case 168:
#line 561 "bi/parser.ypp"
                                                                                                          { push_raw(); }
#line 2366 "bi/parser.cpp"
    break;

  case 169:
#line 561 "bi/parser.ypp"
                                                                                                                                                 { ((*yyvalp).valStatement) = new bi::Class((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-9)].yystate.yysemantics.yysval.valAnnotation), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-7)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), false, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2372 "bi/parser.cpp"
    break;

  case 170:
#line 562 "bi/parser.ypp"
                                                                        { push_raw(); }
#line 2378 "bi/parser.cpp"
    break;

  case 171:
#line 562 "bi/parser.ypp"
                                                                                                                                                 { ((*yyvalp).valStatement) = new bi::Class((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valAnnotation), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), empty_type((*yylocp)), false, empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2384 "bi/parser.cpp"
    break;

  case 172:
#line 563 "bi/parser.ypp"
                                                                   { push_raw(); }
#line 2390 "bi/parser.cpp"
    break;

  case 173:
#line 563 "bi/parser.ypp"
                                                                                                                                                 { ((*yyvalp).valStatement) = new bi::Class((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-7)].yystate.yysemantics.yysval.valAnnotation), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), true, empty_expr((*yylocp)), empty_stmt((*yylocp)), make_doc_loc((*yylocp))); }
#line 2396 "bi/parser.cpp"
    break;

  case 174:
#line 567 "bi/parser.ypp"
                                    { push_raw(); ((*yyvalp).valStatement) = new bi::Basic(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), false, make_doc_loc((*yylocp))); }
#line 2402 "bi/parser.cpp"
    break;

  case 175:
#line 568 "bi/parser.ypp"
                                    { push_raw(); ((*yyvalp).valStatement) = new bi::Basic(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), true, make_doc_loc((*yylocp))); }
#line 2408 "bi/parser.cpp"
    break;

  case 176:
#line 569 "bi/parser.ypp"
                                    { push_raw(); ((*yyvalp).valStatement) = new bi::Basic(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), empty_expr((*yylocp)), empty_type((*yylocp)), false, make_doc_loc((*yylocp))); }
#line 2414 "bi/parser.cpp"
    break;

  case 177:
#line 573 "bi/parser.ypp"
                         { push_raw(); ((*yyvalp).valStatement) = new bi::Raw(new bi::Name("cpp"), pop_raw(), make_loc((*yylocp))); }
#line 2420 "bi/parser.cpp"
    break;

  case 178:
#line 577 "bi/parser.ypp"
                         { push_raw(); ((*yyvalp).valStatement) = new bi::Raw(new bi::Name("hpp"), pop_raw(), make_loc((*yylocp))); }
#line 2426 "bi/parser.cpp"
    break;

  case 179:
#line 581 "bi/parser.ypp"
                      { ((*yyvalp).valName) = new bi::Name("~"); }
#line 2432 "bi/parser.cpp"
    break;

  case 180:
#line 582 "bi/parser.ypp"
                      { ((*yyvalp).valName) = new bi::Name("<~"); }
#line 2438 "bi/parser.cpp"
    break;

  case 181:
#line 583 "bi/parser.ypp"
                      { ((*yyvalp).valName) = new bi::Name("~>"); }
#line 2444 "bi/parser.cpp"
    break;

  case 182:
#line 584 "bi/parser.ypp"
                      { ((*yyvalp).valName) = new bi::Name("<-?"); }
#line 2450 "bi/parser.cpp"
    break;

  case 183:
#line 588 "bi/parser.ypp"
                                                 { ((*yyvalp).valStatement) = new bi::Assume((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2456 "bi/parser.cpp"
    break;

  case 184:
#line 592 "bi/parser.ypp"
                      { ((*yyvalp).valStatement) = new bi::ExpressionStatement((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2462 "bi/parser.cpp"
    break;

  case 185:
#line 596 "bi/parser.ypp"
                                        { ((*yyvalp).valStatement) = new bi::If((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valStatement), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2468 "bi/parser.cpp"
    break;

  case 186:
#line 597 "bi/parser.ypp"
                                        { ((*yyvalp).valStatement) = new bi::If((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valStatement), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2474 "bi/parser.cpp"
    break;

  case 187:
#line 598 "bi/parser.ypp"
                                        { ((*yyvalp).valStatement) = new bi::If((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), empty_stmt((*yylocp)), make_loc((*yylocp))); }
#line 2480 "bi/parser.cpp"
    break;

  case 188:
#line 602 "bi/parser.ypp"
                          { ((*yyvalp).valStatement) = new bi::LocalVariable((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valName), new bi::NamedType(new bi::Name("Integer")), make_loc((*yylocp))); }
#line 2486 "bi/parser.cpp"
    break;

  case 189:
#line 606 "bi/parser.ypp"
                                                                             { ((*yyvalp).valStatement) = new bi::For(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valStatement), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2492 "bi/parser.cpp"
    break;

  case 190:
#line 610 "bi/parser.ypp"
               { ((*yyvalp).valAnnotation) = bi::DYNAMIC; }
#line 2498 "bi/parser.cpp"
    break;

  case 191:
#line 616 "bi/parser.ypp"
                                                                                                          { ((*yyvalp).valStatement) = new bi::Parallel((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-8)].yystate.yysemantics.yysval.valAnnotation), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valStatement), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2504 "bi/parser.cpp"
    break;

  case 192:
#line 617 "bi/parser.ypp"
                                                                                                          { ((*yyvalp).valStatement) = new bi::Parallel(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valStatement), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2510 "bi/parser.cpp"
    break;

  case 193:
#line 621 "bi/parser.ypp"
                               { ((*yyvalp).valStatement) = new bi::While((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2516 "bi/parser.cpp"
    break;

  case 194:
#line 625 "bi/parser.ypp"
                                      { ((*yyvalp).valStatement) = new bi::DoWhile((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valStatement), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2522 "bi/parser.cpp"
    break;

  case 195:
#line 629 "bi/parser.ypp"
              { ((*yyvalp).valStatement) = new bi::Block((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2528 "bi/parser.cpp"
    break;

  case 196:
#line 633 "bi/parser.ypp"
                             { ((*yyvalp).valStatement) = new bi::Assert((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2534 "bi/parser.cpp"
    break;

  case 197:
#line 637 "bi/parser.ypp"
                                      { ((*yyvalp).valStatement) = new bi::Return((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2540 "bi/parser.cpp"
    break;

  case 198:
#line 641 "bi/parser.ypp"
                            { ((*yyvalp).valStatement) = new bi::Yield((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2546 "bi/parser.cpp"
    break;

  case 213:
#line 662 "bi/parser.ypp"
                            { ((*yyvalp).valStatement) = new bi::StatementList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2552 "bi/parser.cpp"
    break;

  case 215:
#line 667 "bi/parser.ypp"
                  { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2558 "bi/parser.cpp"
    break;

  case 223:
#line 681 "bi/parser.ypp"
                                        { ((*yyvalp).valStatement) = new bi::StatementList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2564 "bi/parser.cpp"
    break;

  case 225:
#line 686 "bi/parser.ypp"
                        { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2570 "bi/parser.cpp"
    break;

  case 237:
#line 704 "bi/parser.ypp"
                                      { ((*yyvalp).valStatement) = new bi::StatementList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2576 "bi/parser.cpp"
    break;

  case 239:
#line 709 "bi/parser.ypp"
                       { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2582 "bi/parser.cpp"
    break;

  case 240:
#line 713 "bi/parser.ypp"
                                { compiler->setRoot((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement)); }
#line 2588 "bi/parser.cpp"
    break;

  case 241:
#line 717 "bi/parser.ypp"
                     { ((*yyvalp).valType) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType); }
#line 2594 "bi/parser.cpp"
    break;

  case 243:
#line 722 "bi/parser.ypp"
                   { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 2600 "bi/parser.cpp"
    break;

  case 244:
#line 726 "bi/parser.ypp"
                                      { ((*yyvalp).valType) = new bi::FiberType(empty_type((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2606 "bi/parser.cpp"
    break;

  case 245:
#line 727 "bi/parser.ypp"
                                      { ((*yyvalp).valType) = new bi::FiberType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2612 "bi/parser.cpp"
    break;

  case 246:
#line 728 "bi/parser.ypp"
                                      { ((*yyvalp).valType) = new bi::FiberType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_type((*yylocp)), make_loc((*yylocp))); }
#line 2618 "bi/parser.cpp"
    break;

  case 248:
#line 733 "bi/parser.ypp"
                         { ((*yyvalp).valType) = new bi::FiberType(empty_type((*yylocp)), empty_type((*yylocp)), make_loc((*yylocp))); }
#line 2624 "bi/parser.cpp"
    break;

  case 249:
#line 737 "bi/parser.ypp"
                          { ((*yyvalp).valExpression) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression); }
#line 2630 "bi/parser.cpp"
    break;

  case 251:
#line 742 "bi/parser.ypp"
             { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 2636 "bi/parser.cpp"
    break;

  case 252:
#line 746 "bi/parser.ypp"
                                   { ((*yyvalp).valStatement) = new bi::Braces((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2642 "bi/parser.cpp"
    break;

  case 254:
#line 751 "bi/parser.ypp"
              { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2648 "bi/parser.cpp"
    break;

  case 255:
#line 755 "bi/parser.ypp"
                                         { ((*yyvalp).valStatement) = new bi::Braces((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2654 "bi/parser.cpp"
    break;

  case 257:
#line 760 "bi/parser.ypp"
                    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2660 "bi/parser.cpp"
    break;

  case 259:
#line 773 "bi/parser.ypp"
                                           { ((*yyvalp).valType) = new bi::NamedType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2666 "bi/parser.cpp"
    break;

  case 260:
#line 774 "bi/parser.ypp"
                                           { yywarn("using weak pointers is no longer necessary"); ((*yyvalp).valType) = new bi::NamedType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2672 "bi/parser.cpp"
    break;

  case 262:
#line 779 "bi/parser.ypp"
                         { ((*yyvalp).valType) = new bi::TupleType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2678 "bi/parser.cpp"
    break;

  case 264:
#line 784 "bi/parser.ypp"
                                                            { ((*yyvalp).valType) = new bi::OptionalType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2684 "bi/parser.cpp"
    break;

  case 265:
#line 785 "bi/parser.ypp"
                                                            { ((*yyvalp).valType) = new bi::FiberType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2690 "bi/parser.cpp"
    break;

  case 266:
#line 786 "bi/parser.ypp"
                                                            { ((*yyvalp).valType) = new bi::FiberType(empty_type((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2696 "bi/parser.cpp"
    break;

  case 267:
#line 787 "bi/parser.ypp"
                                                            { ((*yyvalp).valType) = new bi::FiberType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_type((*yylocp)), make_loc((*yylocp))); }
#line 2702 "bi/parser.cpp"
    break;

  case 268:
#line 788 "bi/parser.ypp"
                                                            { ((*yyvalp).valType) = new bi::FiberType(empty_type((*yylocp)), empty_type((*yylocp)), make_loc((*yylocp))); }
#line 2708 "bi/parser.cpp"
    break;

  case 269:
#line 789 "bi/parser.ypp"
                                                            { ((*yyvalp).valType) = new bi::MemberType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2714 "bi/parser.cpp"
    break;

  case 270:
#line 790 "bi/parser.ypp"
                                                            { ((*yyvalp).valType) = new bi::ArrayType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valInt), make_loc((*yylocp))); }
#line 2720 "bi/parser.cpp"
    break;

  case 271:
#line 791 "bi/parser.ypp"
                                                            { ((*yyvalp).valType) = new bi::FunctionType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2726 "bi/parser.cpp"
    break;

  case 273:
#line 796 "bi/parser.ypp"
                          { ((*yyvalp).valType) = new bi::TypeList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2732 "bi/parser.cpp"
    break;

  case 275:
#line 801 "bi/parser.ypp"
                 { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 2738 "bi/parser.cpp"
    break;


#line 2742 "bi/parser.cpp"

      default: break;
    }

  return yyok;
# undef yyerrok
# undef YYABORT
# undef YYACCEPT
# undef YYERROR
# undef YYBACKUP
# undef yyclearin
# undef YYRECOVERING
}


static void
yyuserMerge (int yyn, YYSTYPE* yy0, YYSTYPE* yy1)
{
  YYUSE (yy0);
  YYUSE (yy1);

  switch (yyn)
    {

      default: break;
    }
}

                              /* Bison grammar-table manipulation.  */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep, YYLTYPE *yylocationp)
{
  YYUSE (yyvaluep);
  YYUSE (yylocationp);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}

/** Number of symbols composing the right hand side of rule #RULE.  */
static inline int
yyrhsLength (yyRuleNum yyrule)
{
  return yyr2[yyrule];
}

static void
yydestroyGLRState (char const *yymsg, yyGLRState *yys)
{
  if (yys->yyresolved)
    yydestruct (yymsg, yystos[yys->yylrState],
                &yys->yysemantics.yysval, &yys->yyloc);
  else
    {
#if YYDEBUG
      if (yydebug)
        {
          if (yys->yysemantics.yyfirstVal)
            YY_FPRINTF ((stderr, "%s unresolved", yymsg));
          else
            YY_FPRINTF ((stderr, "%s incomplete", yymsg));
          YY_SYMBOL_PRINT ("", yystos[yys->yylrState], YY_NULLPTR, &yys->yyloc);
        }
#endif

      if (yys->yysemantics.yyfirstVal)
        {
          yySemanticOption *yyoption = yys->yysemantics.yyfirstVal;
          yyGLRState *yyrh;
          int yyn;
          for (yyrh = yyoption->yystate, yyn = yyrhsLength (yyoption->yyrule);
               yyn > 0;
               yyrh = yyrh->yypred, yyn -= 1)
            yydestroyGLRState (yymsg, yyrh);
        }
    }
}

/** Left-hand-side symbol for rule #YYRULE.  */
static inline yySymbol
yylhsNonterm (yyRuleNum yyrule)
{
  return yyr1[yyrule];
}

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

/** True iff LR state YYSTATE has only a default reduction (regardless
 *  of token).  */
static inline yybool
yyisDefaultedState (yyStateNum yystate)
{
  return yypact_value_is_default (yypact[yystate]);
}

/** The default reduction for YYSTATE, assuming it has one.  */
static inline yyRuleNum
yydefaultAction (yyStateNum yystate)
{
  return yydefact[yystate];
}

#define yytable_value_is_error(Yyn) \
  0

/** The action to take in YYSTATE on seeing YYTOKEN.
 *  Result R means
 *    R < 0:  Reduce on rule -R.
 *    R = 0:  Error.
 *    R > 0:  Shift to state R.
 *  Set *YYCONFLICTS to a pointer into yyconfl to a 0-terminated list
 *  of conflicting reductions.
 */
static inline int
yygetLRActions (yyStateNum yystate, yySymbol yytoken, const short** yyconflicts)
{
  int yyindex = yypact[yystate] + yytoken;
  if (yyisDefaultedState (yystate)
      || yyindex < 0 || YYLAST < yyindex || yycheck[yyindex] != yytoken)
    {
      *yyconflicts = yyconfl;
      return -yydefact[yystate];
    }
  else if (! yytable_value_is_error (yytable[yyindex]))
    {
      *yyconflicts = yyconfl + yyconflp[yyindex];
      return yytable[yyindex];
    }
  else
    {
      *yyconflicts = yyconfl + yyconflp[yyindex];
      return 0;
    }
}

/** Compute post-reduction state.
 * \param yystate   the current state
 * \param yysym     the nonterminal to push on the stack
 */
static inline yyStateNum
yyLRgotoState (yyStateNum yystate, yySymbol yysym)
{
  int yyr = yypgoto[yysym - YYNTOKENS] + yystate;
  if (0 <= yyr && yyr <= YYLAST && yycheck[yyr] == yystate)
    return yytable[yyr];
  else
    return yydefgoto[yysym - YYNTOKENS];
}

static inline yybool
yyisShiftAction (int yyaction)
{
  return 0 < yyaction;
}

static inline yybool
yyisErrorAction (int yyaction)
{
  return yyaction == 0;
}

                                /* GLRStates */

/** Return a fresh GLRStackItem in YYSTACKP.  The item is an LR state
 *  if YYISSTATE, and otherwise a semantic option.  Callers should call
 *  YY_RESERVE_GLRSTACK afterwards to make sure there is sufficient
 *  headroom.  */

static inline yyGLRStackItem*
yynewGLRStackItem (yyGLRStack* yystackp, yybool yyisState)
{
  yyGLRStackItem* yynewItem = yystackp->yynextFree;
  yystackp->yyspaceLeft -= 1;
  yystackp->yynextFree += 1;
  yynewItem->yystate.yyisState = yyisState;
  return yynewItem;
}

/** Add a new semantic action that will execute the action for rule
 *  YYRULE on the semantic values in YYRHS to the list of
 *  alternative actions for YYSTATE.  Assumes that YYRHS comes from
 *  stack #YYK of *YYSTACKP. */
static void
yyaddDeferredAction (yyGLRStack* yystackp, ptrdiff_t yyk, yyGLRState* yystate,
                     yyGLRState* yyrhs, yyRuleNum yyrule)
{
  yySemanticOption* yynewOption =
    &yynewGLRStackItem (yystackp, yyfalse)->yyoption;
  YY_ASSERT (!yynewOption->yyisState);
  yynewOption->yystate = yyrhs;
  yynewOption->yyrule = yyrule;
  if (yystackp->yytops.yylookaheadNeeds[yyk])
    {
      yynewOption->yyrawchar = yychar;
      yynewOption->yyval = yylval;
      yynewOption->yyloc = yylloc;
    }
  else
    yynewOption->yyrawchar = YYEMPTY;
  yynewOption->yynext = yystate->yysemantics.yyfirstVal;
  yystate->yysemantics.yyfirstVal = yynewOption;

  YY_RESERVE_GLRSTACK (yystackp);
}

                                /* GLRStacks */

/** Initialize YYSET to a singleton set containing an empty stack.  */
static yybool
yyinitStateSet (yyGLRStateSet* yyset)
{
  yyset->yysize = 1;
  yyset->yycapacity = 16;
  yyset->yystates
    = YY_CAST (yyGLRState**,
               YYMALLOC (YY_CAST (size_t, yyset->yycapacity)
                         * sizeof yyset->yystates[0]));
  if (! yyset->yystates)
    return yyfalse;
  yyset->yystates[0] = YY_NULLPTR;
  yyset->yylookaheadNeeds
    = YY_CAST (yybool*,
               YYMALLOC (YY_CAST (size_t, yyset->yycapacity)
                         * sizeof yyset->yylookaheadNeeds[0]));
  if (! yyset->yylookaheadNeeds)
    {
      YYFREE (yyset->yystates);
      return yyfalse;
    }
  memset (yyset->yylookaheadNeeds,
          0,
          YY_CAST (size_t, yyset->yycapacity) * sizeof yyset->yylookaheadNeeds[0]);
  return yytrue;
}

static void yyfreeStateSet (yyGLRStateSet* yyset)
{
  YYFREE (yyset->yystates);
  YYFREE (yyset->yylookaheadNeeds);
}

/** Initialize *YYSTACKP to a single empty stack, with total maximum
 *  capacity for all stacks of YYSIZE.  */
static yybool
yyinitGLRStack (yyGLRStack* yystackp, ptrdiff_t yysize)
{
  yystackp->yyerrState = 0;
  yynerrs = 0;
  yystackp->yyspaceLeft = yysize;
  yystackp->yyitems
    = YY_CAST (yyGLRStackItem*,
               YYMALLOC (YY_CAST (size_t, yysize)
                         * sizeof yystackp->yynextFree[0]));
  if (!yystackp->yyitems)
    return yyfalse;
  yystackp->yynextFree = yystackp->yyitems;
  yystackp->yysplitPoint = YY_NULLPTR;
  yystackp->yylastDeleted = YY_NULLPTR;
  return yyinitStateSet (&yystackp->yytops);
}


#if YYSTACKEXPANDABLE
# define YYRELOC(YYFROMITEMS, YYTOITEMS, YYX, YYTYPE)                   \
  &((YYTOITEMS)                                                         \
    - ((YYFROMITEMS) - YY_REINTERPRET_CAST (yyGLRStackItem*, (YYX))))->YYTYPE

/** If *YYSTACKP is expandable, extend it.  WARNING: Pointers into the
    stack from outside should be considered invalid after this call.
    We always expand when there are 1 or fewer items left AFTER an
    allocation, so that we can avoid having external pointers exist
    across an allocation.  */
static void
yyexpandGLRStack (yyGLRStack* yystackp)
{
  yyGLRStackItem* yynewItems;
  yyGLRStackItem* yyp0, *yyp1;
  ptrdiff_t yynewSize;
  ptrdiff_t yyn;
  ptrdiff_t yysize = yystackp->yynextFree - yystackp->yyitems;
  if (YYMAXDEPTH - YYHEADROOM < yysize)
    yyMemoryExhausted (yystackp);
  yynewSize = 2*yysize;
  if (YYMAXDEPTH < yynewSize)
    yynewSize = YYMAXDEPTH;
  yynewItems
    = YY_CAST (yyGLRStackItem*,
               YYMALLOC (YY_CAST (size_t, yynewSize)
                         * sizeof yynewItems[0]));
  if (! yynewItems)
    yyMemoryExhausted (yystackp);
  for (yyp0 = yystackp->yyitems, yyp1 = yynewItems, yyn = yysize;
       0 < yyn;
       yyn -= 1, yyp0 += 1, yyp1 += 1)
    {
      *yyp1 = *yyp0;
      if (*YY_REINTERPRET_CAST (yybool *, yyp0))
        {
          yyGLRState* yys0 = &yyp0->yystate;
          yyGLRState* yys1 = &yyp1->yystate;
          if (yys0->yypred != YY_NULLPTR)
            yys1->yypred =
              YYRELOC (yyp0, yyp1, yys0->yypred, yystate);
          if (! yys0->yyresolved && yys0->yysemantics.yyfirstVal != YY_NULLPTR)
            yys1->yysemantics.yyfirstVal =
              YYRELOC (yyp0, yyp1, yys0->yysemantics.yyfirstVal, yyoption);
        }
      else
        {
          yySemanticOption* yyv0 = &yyp0->yyoption;
          yySemanticOption* yyv1 = &yyp1->yyoption;
          if (yyv0->yystate != YY_NULLPTR)
            yyv1->yystate = YYRELOC (yyp0, yyp1, yyv0->yystate, yystate);
          if (yyv0->yynext != YY_NULLPTR)
            yyv1->yynext = YYRELOC (yyp0, yyp1, yyv0->yynext, yyoption);
        }
    }
  if (yystackp->yysplitPoint != YY_NULLPTR)
    yystackp->yysplitPoint = YYRELOC (yystackp->yyitems, yynewItems,
                                      yystackp->yysplitPoint, yystate);

  for (yyn = 0; yyn < yystackp->yytops.yysize; yyn += 1)
    if (yystackp->yytops.yystates[yyn] != YY_NULLPTR)
      yystackp->yytops.yystates[yyn] =
        YYRELOC (yystackp->yyitems, yynewItems,
                 yystackp->yytops.yystates[yyn], yystate);
  YYFREE (yystackp->yyitems);
  yystackp->yyitems = yynewItems;
  yystackp->yynextFree = yynewItems + yysize;
  yystackp->yyspaceLeft = yynewSize - yysize;
}
#endif

static void
yyfreeGLRStack (yyGLRStack* yystackp)
{
  YYFREE (yystackp->yyitems);
  yyfreeStateSet (&yystackp->yytops);
}

/** Assuming that YYS is a GLRState somewhere on *YYSTACKP, update the
 *  splitpoint of *YYSTACKP, if needed, so that it is at least as deep as
 *  YYS.  */
static inline void
yyupdateSplit (yyGLRStack* yystackp, yyGLRState* yys)
{
  if (yystackp->yysplitPoint != YY_NULLPTR && yystackp->yysplitPoint > yys)
    yystackp->yysplitPoint = yys;
}

/** Invalidate stack #YYK in *YYSTACKP.  */
static inline void
yymarkStackDeleted (yyGLRStack* yystackp, ptrdiff_t yyk)
{
  if (yystackp->yytops.yystates[yyk] != YY_NULLPTR)
    yystackp->yylastDeleted = yystackp->yytops.yystates[yyk];
  yystackp->yytops.yystates[yyk] = YY_NULLPTR;
}

/** Undelete the last stack in *YYSTACKP that was marked as deleted.  Can
    only be done once after a deletion, and only when all other stacks have
    been deleted.  */
static void
yyundeleteLastStack (yyGLRStack* yystackp)
{
  if (yystackp->yylastDeleted == YY_NULLPTR || yystackp->yytops.yysize != 0)
    return;
  yystackp->yytops.yystates[0] = yystackp->yylastDeleted;
  yystackp->yytops.yysize = 1;
  YY_DPRINTF ((stderr, "Restoring last deleted stack as stack #0.\n"));
  yystackp->yylastDeleted = YY_NULLPTR;
}

static inline void
yyremoveDeletes (yyGLRStack* yystackp)
{
  ptrdiff_t yyi, yyj;
  yyi = yyj = 0;
  while (yyj < yystackp->yytops.yysize)
    {
      if (yystackp->yytops.yystates[yyi] == YY_NULLPTR)
        {
          if (yyi == yyj)
            YY_DPRINTF ((stderr, "Removing dead stacks.\n"));
          yystackp->yytops.yysize -= 1;
        }
      else
        {
          yystackp->yytops.yystates[yyj] = yystackp->yytops.yystates[yyi];
          /* In the current implementation, it's unnecessary to copy
             yystackp->yytops.yylookaheadNeeds[yyi] since, after
             yyremoveDeletes returns, the parser immediately either enters
             deterministic operation or shifts a token.  However, it doesn't
             hurt, and the code might evolve to need it.  */
          yystackp->yytops.yylookaheadNeeds[yyj] =
            yystackp->yytops.yylookaheadNeeds[yyi];
          if (yyj != yyi)
            YY_DPRINTF ((stderr, "Rename stack %ld -> %ld.\n",
                        YY_CAST (long, yyi), YY_CAST (long, yyj)));
          yyj += 1;
        }
      yyi += 1;
    }
}

/** Shift to a new state on stack #YYK of *YYSTACKP, corresponding to LR
 * state YYLRSTATE, at input position YYPOSN, with (resolved) semantic
 * value *YYVALP and source location *YYLOCP.  */
static inline void
yyglrShift (yyGLRStack* yystackp, ptrdiff_t yyk, yyStateNum yylrState,
            ptrdiff_t yyposn,
            YYSTYPE* yyvalp, YYLTYPE* yylocp)
{
  yyGLRState* yynewState = &yynewGLRStackItem (yystackp, yytrue)->yystate;

  yynewState->yylrState = yylrState;
  yynewState->yyposn = yyposn;
  yynewState->yyresolved = yytrue;
  yynewState->yypred = yystackp->yytops.yystates[yyk];
  yynewState->yysemantics.yysval = *yyvalp;
  yynewState->yyloc = *yylocp;
  yystackp->yytops.yystates[yyk] = yynewState;

  YY_RESERVE_GLRSTACK (yystackp);
}

/** Shift stack #YYK of *YYSTACKP, to a new state corresponding to LR
 *  state YYLRSTATE, at input position YYPOSN, with the (unresolved)
 *  semantic value of YYRHS under the action for YYRULE.  */
static inline void
yyglrShiftDefer (yyGLRStack* yystackp, ptrdiff_t yyk, yyStateNum yylrState,
                 ptrdiff_t yyposn, yyGLRState* yyrhs, yyRuleNum yyrule)
{
  yyGLRState* yynewState = &yynewGLRStackItem (yystackp, yytrue)->yystate;
  YY_ASSERT (yynewState->yyisState);

  yynewState->yylrState = yylrState;
  yynewState->yyposn = yyposn;
  yynewState->yyresolved = yyfalse;
  yynewState->yypred = yystackp->yytops.yystates[yyk];
  yynewState->yysemantics.yyfirstVal = YY_NULLPTR;
  yystackp->yytops.yystates[yyk] = yynewState;

  /* Invokes YY_RESERVE_GLRSTACK.  */
  yyaddDeferredAction (yystackp, yyk, yynewState, yyrhs, yyrule);
}

#if !YYDEBUG
# define YY_REDUCE_PRINT(Args)
#else
# define YY_REDUCE_PRINT(Args)          \
  do {                                  \
    if (yydebug)                        \
      yy_reduce_print Args;             \
  } while (0)

/*----------------------------------------------------------------------.
| Report that stack #YYK of *YYSTACKP is going to be reduced by YYRULE. |
`----------------------------------------------------------------------*/

static inline void
yy_reduce_print (yybool yynormal, yyGLRStackItem* yyvsp, ptrdiff_t yyk,
                 yyRuleNum yyrule)
{
  int yynrhs = yyrhsLength (yyrule);
  int yylow = 1;
  int yyi;
  YY_FPRINTF ((stderr, "Reducing stack %ld by rule %d (line %d):\n",
               YY_CAST (long, yyk), yyrule - 1, yyrline[yyrule]));
  if (! yynormal)
    yyfillin (yyvsp, 1, -yynrhs);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YY_FPRINTF ((stderr, "   $%d = ", yyi + 1));
      yy_symbol_print (stderr,
                       yystos[yyvsp[yyi - yynrhs + 1].yystate.yylrState],
                       &yyvsp[yyi - yynrhs + 1].yystate.yysemantics.yysval,
                       &(YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL ((yyi + 1) - (yynrhs))].yystate.yyloc)                       );
      if (!yyvsp[yyi - yynrhs + 1].yystate.yyresolved)
        YY_FPRINTF ((stderr, " (unresolved)"));
      YY_FPRINTF ((stderr, "\n"));
    }
}
#endif

/** Pop the symbols consumed by reduction #YYRULE from the top of stack
 *  #YYK of *YYSTACKP, and perform the appropriate semantic action on their
 *  semantic values.  Assumes that all ambiguities in semantic values
 *  have been previously resolved.  Set *YYVALP to the resulting value,
 *  and *YYLOCP to the computed location (if any).  Return value is as
 *  for userAction.  */
static inline YYRESULTTAG
yydoAction (yyGLRStack* yystackp, ptrdiff_t yyk, yyRuleNum yyrule,
            YYSTYPE* yyvalp, YYLTYPE *yylocp)
{
  int yynrhs = yyrhsLength (yyrule);

  if (yystackp->yysplitPoint == YY_NULLPTR)
    {
      /* Standard special case: single stack.  */
      yyGLRStackItem* yyrhs
        = YY_REINTERPRET_CAST (yyGLRStackItem*, yystackp->yytops.yystates[yyk]);
      YY_ASSERT (yyk == 0);
      yystackp->yynextFree -= yynrhs;
      yystackp->yyspaceLeft += yynrhs;
      yystackp->yytops.yystates[0] = & yystackp->yynextFree[-1].yystate;
      YY_REDUCE_PRINT ((yytrue, yyrhs, yyk, yyrule));
      return yyuserAction (yyrule, yynrhs, yyrhs, yystackp,
                           yyvalp, yylocp);
    }
  else
    {
      yyGLRStackItem yyrhsVals[YYMAXRHS + YYMAXLEFT + 1];
      yyGLRState* yys = yyrhsVals[YYMAXRHS + YYMAXLEFT].yystate.yypred
        = yystackp->yytops.yystates[yyk];
      int yyi;
      if (yynrhs == 0)
        /* Set default location.  */
        yyrhsVals[YYMAXRHS + YYMAXLEFT - 1].yystate.yyloc = yys->yyloc;
      for (yyi = 0; yyi < yynrhs; yyi += 1)
        {
          yys = yys->yypred;
          YY_ASSERT (yys);
        }
      yyupdateSplit (yystackp, yys);
      yystackp->yytops.yystates[yyk] = yys;
      YY_REDUCE_PRINT ((yyfalse, yyrhsVals + YYMAXRHS + YYMAXLEFT - 1, yyk, yyrule));
      return yyuserAction (yyrule, yynrhs, yyrhsVals + YYMAXRHS + YYMAXLEFT - 1,
                           yystackp, yyvalp, yylocp);
    }
}

/** Pop items off stack #YYK of *YYSTACKP according to grammar rule YYRULE,
 *  and push back on the resulting nonterminal symbol.  Perform the
 *  semantic action associated with YYRULE and store its value with the
 *  newly pushed state, if YYFORCEEVAL or if *YYSTACKP is currently
 *  unambiguous.  Otherwise, store the deferred semantic action with
 *  the new state.  If the new state would have an identical input
 *  position, LR state, and predecessor to an existing state on the stack,
 *  it is identified with that existing state, eliminating stack #YYK from
 *  *YYSTACKP.  In this case, the semantic value is
 *  added to the options for the existing state's semantic value.
 */
static inline YYRESULTTAG
yyglrReduce (yyGLRStack* yystackp, ptrdiff_t yyk, yyRuleNum yyrule,
             yybool yyforceEval)
{
  ptrdiff_t yyposn = yystackp->yytops.yystates[yyk]->yyposn;

  if (yyforceEval || yystackp->yysplitPoint == YY_NULLPTR)
    {
      YYSTYPE yysval;
      YYLTYPE yyloc;

      YYRESULTTAG yyflag = yydoAction (yystackp, yyk, yyrule, &yysval, &yyloc);
      if (yyflag == yyerr && yystackp->yysplitPoint != YY_NULLPTR)
        YY_DPRINTF ((stderr,
                     "Parse on stack %ld rejected by rule %d (line %d).\n",
                     YY_CAST (long, yyk), yyrule - 1, yyrline[yyrule - 1]));
      if (yyflag != yyok)
        return yyflag;
      YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyrule], &yysval, &yyloc);
      yyglrShift (yystackp, yyk,
                  yyLRgotoState (yystackp->yytops.yystates[yyk]->yylrState,
                                 yylhsNonterm (yyrule)),
                  yyposn, &yysval, &yyloc);
    }
  else
    {
      ptrdiff_t yyi;
      int yyn;
      yyGLRState* yys, *yys0 = yystackp->yytops.yystates[yyk];
      yyStateNum yynewLRState;

      for (yys = yystackp->yytops.yystates[yyk], yyn = yyrhsLength (yyrule);
           0 < yyn; yyn -= 1)
        {
          yys = yys->yypred;
          YY_ASSERT (yys);
        }
      yyupdateSplit (yystackp, yys);
      yynewLRState = yyLRgotoState (yys->yylrState, yylhsNonterm (yyrule));
      YY_DPRINTF ((stderr,
                   "Reduced stack %ld by rule %d (line %d); action deferred.  "
                   "Now in state %d.\n",
                   YY_CAST (long, yyk), yyrule - 1, yyrline[yyrule - 1],
                   yynewLRState));
      for (yyi = 0; yyi < yystackp->yytops.yysize; yyi += 1)
        if (yyi != yyk && yystackp->yytops.yystates[yyi] != YY_NULLPTR)
          {
            yyGLRState *yysplit = yystackp->yysplitPoint;
            yyGLRState *yyp = yystackp->yytops.yystates[yyi];
            while (yyp != yys && yyp != yysplit && yyp->yyposn >= yyposn)
              {
                if (yyp->yylrState == yynewLRState && yyp->yypred == yys)
                  {
                    yyaddDeferredAction (yystackp, yyk, yyp, yys0, yyrule);
                    yymarkStackDeleted (yystackp, yyk);
                    YY_DPRINTF ((stderr, "Merging stack %ld into stack %ld.\n",
                                 YY_CAST (long, yyk), YY_CAST (long, yyi)));
                    return yyok;
                  }
                yyp = yyp->yypred;
              }
          }
      yystackp->yytops.yystates[yyk] = yys;
      yyglrShiftDefer (yystackp, yyk, yynewLRState, yyposn, yys0, yyrule);
    }
  return yyok;
}

static ptrdiff_t
yysplitStack (yyGLRStack* yystackp, ptrdiff_t yyk)
{
  if (yystackp->yysplitPoint == YY_NULLPTR)
    {
      YY_ASSERT (yyk == 0);
      yystackp->yysplitPoint = yystackp->yytops.yystates[yyk];
    }
  if (yystackp->yytops.yycapacity <= yystackp->yytops.yysize)
    {
      ptrdiff_t state_size = sizeof yystackp->yytops.yystates[0];
      ptrdiff_t half_max_capacity = YYSIZEMAX / 2 / state_size;
      if (half_max_capacity < yystackp->yytops.yycapacity)
        yyMemoryExhausted (yystackp);
      yystackp->yytops.yycapacity *= 2;

      {
        yyGLRState** yynewStates
          = YY_CAST (yyGLRState**,
                     YYREALLOC (yystackp->yytops.yystates,
                                (YY_CAST (size_t, yystackp->yytops.yycapacity)
                                 * sizeof yynewStates[0])));
        if (yynewStates == YY_NULLPTR)
          yyMemoryExhausted (yystackp);
        yystackp->yytops.yystates = yynewStates;
      }

      {
        yybool* yynewLookaheadNeeds
          = YY_CAST (yybool*,
                     YYREALLOC (yystackp->yytops.yylookaheadNeeds,
                                (YY_CAST (size_t, yystackp->yytops.yycapacity)
                                 * sizeof yynewLookaheadNeeds[0])));
        if (yynewLookaheadNeeds == YY_NULLPTR)
          yyMemoryExhausted (yystackp);
        yystackp->yytops.yylookaheadNeeds = yynewLookaheadNeeds;
      }
    }
  yystackp->yytops.yystates[yystackp->yytops.yysize]
    = yystackp->yytops.yystates[yyk];
  yystackp->yytops.yylookaheadNeeds[yystackp->yytops.yysize]
    = yystackp->yytops.yylookaheadNeeds[yyk];
  yystackp->yytops.yysize += 1;
  return yystackp->yytops.yysize - 1;
}

/** True iff YYY0 and YYY1 represent identical options at the top level.
 *  That is, they represent the same rule applied to RHS symbols
 *  that produce the same terminal symbols.  */
static yybool
yyidenticalOptions (yySemanticOption* yyy0, yySemanticOption* yyy1)
{
  if (yyy0->yyrule == yyy1->yyrule)
    {
      yyGLRState *yys0, *yys1;
      int yyn;
      for (yys0 = yyy0->yystate, yys1 = yyy1->yystate,
           yyn = yyrhsLength (yyy0->yyrule);
           yyn > 0;
           yys0 = yys0->yypred, yys1 = yys1->yypred, yyn -= 1)
        if (yys0->yyposn != yys1->yyposn)
          return yyfalse;
      return yytrue;
    }
  else
    return yyfalse;
}

/** Assuming identicalOptions (YYY0,YYY1), destructively merge the
 *  alternative semantic values for the RHS-symbols of YYY1 and YYY0.  */
static void
yymergeOptionSets (yySemanticOption* yyy0, yySemanticOption* yyy1)
{
  yyGLRState *yys0, *yys1;
  int yyn;
  for (yys0 = yyy0->yystate, yys1 = yyy1->yystate,
       yyn = yyrhsLength (yyy0->yyrule);
       0 < yyn;
       yys0 = yys0->yypred, yys1 = yys1->yypred, yyn -= 1)
    {
      if (yys0 == yys1)
        break;
      else if (yys0->yyresolved)
        {
          yys1->yyresolved = yytrue;
          yys1->yysemantics.yysval = yys0->yysemantics.yysval;
        }
      else if (yys1->yyresolved)
        {
          yys0->yyresolved = yytrue;
          yys0->yysemantics.yysval = yys1->yysemantics.yysval;
        }
      else
        {
          yySemanticOption** yyz0p = &yys0->yysemantics.yyfirstVal;
          yySemanticOption* yyz1 = yys1->yysemantics.yyfirstVal;
          while (yytrue)
            {
              if (yyz1 == *yyz0p || yyz1 == YY_NULLPTR)
                break;
              else if (*yyz0p == YY_NULLPTR)
                {
                  *yyz0p = yyz1;
                  break;
                }
              else if (*yyz0p < yyz1)
                {
                  yySemanticOption* yyz = *yyz0p;
                  *yyz0p = yyz1;
                  yyz1 = yyz1->yynext;
                  (*yyz0p)->yynext = yyz;
                }
              yyz0p = &(*yyz0p)->yynext;
            }
          yys1->yysemantics.yyfirstVal = yys0->yysemantics.yyfirstVal;
        }
    }
}

/** Y0 and Y1 represent two possible actions to take in a given
 *  parsing state; return 0 if no combination is possible,
 *  1 if user-mergeable, 2 if Y0 is preferred, 3 if Y1 is preferred.  */
static int
yypreference (yySemanticOption* y0, yySemanticOption* y1)
{
  yyRuleNum r0 = y0->yyrule, r1 = y1->yyrule;
  int p0 = yydprec[r0], p1 = yydprec[r1];

  if (p0 == p1)
    {
      if (yymerger[r0] == 0 || yymerger[r0] != yymerger[r1])
        return 0;
      else
        return 1;
    }
  if (p0 == 0 || p1 == 0)
    return 0;
  if (p0 < p1)
    return 3;
  if (p1 < p0)
    return 2;
  return 0;
}

static YYRESULTTAG yyresolveValue (yyGLRState* yys,
                                   yyGLRStack* yystackp);


/** Resolve the previous YYN states starting at and including state YYS
 *  on *YYSTACKP. If result != yyok, some states may have been left
 *  unresolved possibly with empty semantic option chains.  Regardless
 *  of whether result = yyok, each state has been left with consistent
 *  data so that yydestroyGLRState can be invoked if necessary.  */
static YYRESULTTAG
yyresolveStates (yyGLRState* yys, int yyn,
                 yyGLRStack* yystackp)
{
  if (0 < yyn)
    {
      YY_ASSERT (yys->yypred);
      YYCHK (yyresolveStates (yys->yypred, yyn-1, yystackp));
      if (! yys->yyresolved)
        YYCHK (yyresolveValue (yys, yystackp));
    }
  return yyok;
}

/** Resolve the states for the RHS of YYOPT on *YYSTACKP, perform its
 *  user action, and return the semantic value and location in *YYVALP
 *  and *YYLOCP.  Regardless of whether result = yyok, all RHS states
 *  have been destroyed (assuming the user action destroys all RHS
 *  semantic values if invoked).  */
static YYRESULTTAG
yyresolveAction (yySemanticOption* yyopt, yyGLRStack* yystackp,
                 YYSTYPE* yyvalp, YYLTYPE *yylocp)
{
  yyGLRStackItem yyrhsVals[YYMAXRHS + YYMAXLEFT + 1];
  int yynrhs = yyrhsLength (yyopt->yyrule);
  YYRESULTTAG yyflag =
    yyresolveStates (yyopt->yystate, yynrhs, yystackp);
  if (yyflag != yyok)
    {
      yyGLRState *yys;
      for (yys = yyopt->yystate; yynrhs > 0; yys = yys->yypred, yynrhs -= 1)
        yydestroyGLRState ("Cleanup: popping", yys);
      return yyflag;
    }

  yyrhsVals[YYMAXRHS + YYMAXLEFT].yystate.yypred = yyopt->yystate;
  if (yynrhs == 0)
    /* Set default location.  */
    yyrhsVals[YYMAXRHS + YYMAXLEFT - 1].yystate.yyloc = yyopt->yystate->yyloc;
  {
    int yychar_current = yychar;
    YYSTYPE yylval_current = yylval;
    YYLTYPE yylloc_current = yylloc;
    yychar = yyopt->yyrawchar;
    yylval = yyopt->yyval;
    yylloc = yyopt->yyloc;
    yyflag = yyuserAction (yyopt->yyrule, yynrhs,
                           yyrhsVals + YYMAXRHS + YYMAXLEFT - 1,
                           yystackp, yyvalp, yylocp);
    yychar = yychar_current;
    yylval = yylval_current;
    yylloc = yylloc_current;
  }
  return yyflag;
}

#if YYDEBUG
static void
yyreportTree (yySemanticOption* yyx, int yyindent)
{
  int yynrhs = yyrhsLength (yyx->yyrule);
  int yyi;
  yyGLRState* yys;
  yyGLRState* yystates[1 + YYMAXRHS];
  yyGLRState yyleftmost_state;

  for (yyi = yynrhs, yys = yyx->yystate; 0 < yyi; yyi -= 1, yys = yys->yypred)
    yystates[yyi] = yys;
  if (yys == YY_NULLPTR)
    {
      yyleftmost_state.yyposn = 0;
      yystates[0] = &yyleftmost_state;
    }
  else
    yystates[0] = yys;

  if (yyx->yystate->yyposn < yys->yyposn + 1)
    YY_FPRINTF ((stderr, "%*s%s -> <Rule %d, empty>\n",
                 yyindent, "", yytokenName (yylhsNonterm (yyx->yyrule)),
                 yyx->yyrule - 1));
  else
    YY_FPRINTF ((stderr, "%*s%s -> <Rule %d, tokens %ld .. %ld>\n",
                 yyindent, "", yytokenName (yylhsNonterm (yyx->yyrule)),
                 yyx->yyrule - 1, YY_CAST (long, yys->yyposn + 1),
                 YY_CAST (long, yyx->yystate->yyposn)));
  for (yyi = 1; yyi <= yynrhs; yyi += 1)
    {
      if (yystates[yyi]->yyresolved)
        {
          if (yystates[yyi-1]->yyposn+1 > yystates[yyi]->yyposn)
            YY_FPRINTF ((stderr, "%*s%s <empty>\n", yyindent+2, "",
                         yytokenName (yystos[yystates[yyi]->yylrState])));
          else
            YY_FPRINTF ((stderr, "%*s%s <tokens %ld .. %ld>\n", yyindent+2, "",
                         yytokenName (yystos[yystates[yyi]->yylrState]),
                         YY_CAST (long, yystates[yyi-1]->yyposn + 1),
                         YY_CAST (long, yystates[yyi]->yyposn)));
        }
      else
        yyreportTree (yystates[yyi]->yysemantics.yyfirstVal, yyindent+2);
    }
}
#endif

static YYRESULTTAG
yyreportAmbiguity (yySemanticOption* yyx0,
                   yySemanticOption* yyx1)
{
  YYUSE (yyx0);
  YYUSE (yyx1);

#if YYDEBUG
  YY_FPRINTF ((stderr, "Ambiguity detected.\n"));
  YY_FPRINTF ((stderr, "Option 1,\n"));
  yyreportTree (yyx0, 2);
  YY_FPRINTF ((stderr, "\nOption 2,\n"));
  yyreportTree (yyx1, 2);
  YY_FPRINTF ((stderr, "\n"));
#endif

  yyerror (YY_("syntax is ambiguous"));
  return yyabort;
}

/** Resolve the locations for each of the YYN1 states in *YYSTACKP,
 *  ending at YYS1.  Has no effect on previously resolved states.
 *  The first semantic option of a state is always chosen.  */
static void
yyresolveLocations (yyGLRState *yys1, int yyn1,
                    yyGLRStack *yystackp)
{
  if (0 < yyn1)
    {
      yyresolveLocations (yys1->yypred, yyn1 - 1, yystackp);
      if (!yys1->yyresolved)
        {
          yyGLRStackItem yyrhsloc[1 + YYMAXRHS];
          int yynrhs;
          yySemanticOption *yyoption = yys1->yysemantics.yyfirstVal;
          YY_ASSERT (yyoption);
          yynrhs = yyrhsLength (yyoption->yyrule);
          if (0 < yynrhs)
            {
              yyGLRState *yys;
              int yyn;
              yyresolveLocations (yyoption->yystate, yynrhs,
                                  yystackp);
              for (yys = yyoption->yystate, yyn = yynrhs;
                   yyn > 0;
                   yys = yys->yypred, yyn -= 1)
                yyrhsloc[yyn].yystate.yyloc = yys->yyloc;
            }
          else
            {
              /* Both yyresolveAction and yyresolveLocations traverse the GSS
                 in reverse rightmost order.  It is only necessary to invoke
                 yyresolveLocations on a subforest for which yyresolveAction
                 would have been invoked next had an ambiguity not been
                 detected.  Thus the location of the previous state (but not
                 necessarily the previous state itself) is guaranteed to be
                 resolved already.  */
              yyGLRState *yyprevious = yyoption->yystate;
              yyrhsloc[0].yystate.yyloc = yyprevious->yyloc;
            }
          YYLLOC_DEFAULT ((yys1->yyloc), yyrhsloc, yynrhs);
        }
    }
}

/** Resolve the ambiguity represented in state YYS in *YYSTACKP,
 *  perform the indicated actions, and set the semantic value of YYS.
 *  If result != yyok, the chain of semantic options in YYS has been
 *  cleared instead or it has been left unmodified except that
 *  redundant options may have been removed.  Regardless of whether
 *  result = yyok, YYS has been left with consistent data so that
 *  yydestroyGLRState can be invoked if necessary.  */
static YYRESULTTAG
yyresolveValue (yyGLRState* yys, yyGLRStack* yystackp)
{
  yySemanticOption* yyoptionList = yys->yysemantics.yyfirstVal;
  yySemanticOption* yybest = yyoptionList;
  yySemanticOption** yypp;
  yybool yymerge = yyfalse;
  YYSTYPE yysval;
  YYRESULTTAG yyflag;
  YYLTYPE *yylocp = &yys->yyloc;

  for (yypp = &yyoptionList->yynext; *yypp != YY_NULLPTR; )
    {
      yySemanticOption* yyp = *yypp;

      if (yyidenticalOptions (yybest, yyp))
        {
          yymergeOptionSets (yybest, yyp);
          *yypp = yyp->yynext;
        }
      else
        {
          switch (yypreference (yybest, yyp))
            {
            case 0:
              yyresolveLocations (yys, 1, yystackp);
              return yyreportAmbiguity (yybest, yyp);
              break;
            case 1:
              yymerge = yytrue;
              break;
            case 2:
              break;
            case 3:
              yybest = yyp;
              yymerge = yyfalse;
              break;
            default:
              /* This cannot happen so it is not worth a YY_ASSERT (yyfalse),
                 but some compilers complain if the default case is
                 omitted.  */
              break;
            }
          yypp = &yyp->yynext;
        }
    }

  if (yymerge)
    {
      yySemanticOption* yyp;
      int yyprec = yydprec[yybest->yyrule];
      yyflag = yyresolveAction (yybest, yystackp, &yysval, yylocp);
      if (yyflag == yyok)
        for (yyp = yybest->yynext; yyp != YY_NULLPTR; yyp = yyp->yynext)
          {
            if (yyprec == yydprec[yyp->yyrule])
              {
                YYSTYPE yysval_other;
                YYLTYPE yydummy;
                yyflag = yyresolveAction (yyp, yystackp, &yysval_other, &yydummy);
                if (yyflag != yyok)
                  {
                    yydestruct ("Cleanup: discarding incompletely merged value for",
                                yystos[yys->yylrState],
                                &yysval, yylocp);
                    break;
                  }
                yyuserMerge (yymerger[yyp->yyrule], &yysval, &yysval_other);
              }
          }
    }
  else
    yyflag = yyresolveAction (yybest, yystackp, &yysval, yylocp);

  if (yyflag == yyok)
    {
      yys->yyresolved = yytrue;
      yys->yysemantics.yysval = yysval;
    }
  else
    yys->yysemantics.yyfirstVal = YY_NULLPTR;
  return yyflag;
}

static YYRESULTTAG
yyresolveStack (yyGLRStack* yystackp)
{
  if (yystackp->yysplitPoint != YY_NULLPTR)
    {
      yyGLRState* yys;
      int yyn;

      for (yyn = 0, yys = yystackp->yytops.yystates[0];
           yys != yystackp->yysplitPoint;
           yys = yys->yypred, yyn += 1)
        continue;
      YYCHK (yyresolveStates (yystackp->yytops.yystates[0], yyn, yystackp
                             ));
    }
  return yyok;
}

static void
yycompressStack (yyGLRStack* yystackp)
{
  yyGLRState* yyp, *yyq, *yyr;

  if (yystackp->yytops.yysize != 1 || yystackp->yysplitPoint == YY_NULLPTR)
    return;

  for (yyp = yystackp->yytops.yystates[0], yyq = yyp->yypred, yyr = YY_NULLPTR;
       yyp != yystackp->yysplitPoint;
       yyr = yyp, yyp = yyq, yyq = yyp->yypred)
    yyp->yypred = yyr;

  yystackp->yyspaceLeft += yystackp->yynextFree - yystackp->yyitems;
  yystackp->yynextFree = YY_REINTERPRET_CAST (yyGLRStackItem*, yystackp->yysplitPoint) + 1;
  yystackp->yyspaceLeft -= yystackp->yynextFree - yystackp->yyitems;
  yystackp->yysplitPoint = YY_NULLPTR;
  yystackp->yylastDeleted = YY_NULLPTR;

  while (yyr != YY_NULLPTR)
    {
      yystackp->yynextFree->yystate = *yyr;
      yyr = yyr->yypred;
      yystackp->yynextFree->yystate.yypred = &yystackp->yynextFree[-1].yystate;
      yystackp->yytops.yystates[0] = &yystackp->yynextFree->yystate;
      yystackp->yynextFree += 1;
      yystackp->yyspaceLeft -= 1;
    }
}

static YYRESULTTAG
yyprocessOneStack (yyGLRStack* yystackp, ptrdiff_t yyk,
                   ptrdiff_t yyposn)
{
  while (yystackp->yytops.yystates[yyk] != YY_NULLPTR)
    {
      yyStateNum yystate = yystackp->yytops.yystates[yyk]->yylrState;
      YY_DPRINTF ((stderr, "Stack %ld Entering state %d\n",
                   YY_CAST (long, yyk), yystate));

      YY_ASSERT (yystate != YYFINAL);

      if (yyisDefaultedState (yystate))
        {
          YYRESULTTAG yyflag;
          yyRuleNum yyrule = yydefaultAction (yystate);
          if (yyrule == 0)
            {
              YY_DPRINTF ((stderr, "Stack %ld dies.\n", YY_CAST (long, yyk)));
              yymarkStackDeleted (yystackp, yyk);
              return yyok;
            }
          yyflag = yyglrReduce (yystackp, yyk, yyrule, yyimmediate[yyrule]);
          if (yyflag == yyerr)
            {
              YY_DPRINTF ((stderr,
                           "Stack %ld dies "
                           "(predicate failure or explicit user error).\n",
                           YY_CAST (long, yyk)));
              yymarkStackDeleted (yystackp, yyk);
              return yyok;
            }
          if (yyflag != yyok)
            return yyflag;
        }
      else
        {
          yySymbol yytoken = yygetToken (&yychar);
          const short* yyconflicts;
          const int yyaction = yygetLRActions (yystate, yytoken, &yyconflicts);
          yystackp->yytops.yylookaheadNeeds[yyk] = yytrue;

          while (*yyconflicts != 0)
            {
              YYRESULTTAG yyflag;
              ptrdiff_t yynewStack = yysplitStack (yystackp, yyk);
              YY_DPRINTF ((stderr, "Splitting off stack %ld from %ld.\n",
                           YY_CAST (long, yynewStack), YY_CAST (long, yyk)));
              yyflag = yyglrReduce (yystackp, yynewStack,
                                    *yyconflicts,
                                    yyimmediate[*yyconflicts]);
              if (yyflag == yyok)
                YYCHK (yyprocessOneStack (yystackp, yynewStack,
                                          yyposn));
              else if (yyflag == yyerr)
                {
                  YY_DPRINTF ((stderr, "Stack %ld dies.\n", YY_CAST (long, yynewStack)));
                  yymarkStackDeleted (yystackp, yynewStack);
                }
              else
                return yyflag;
              yyconflicts += 1;
            }

          if (yyisShiftAction (yyaction))
            break;
          else if (yyisErrorAction (yyaction))
            {
              YY_DPRINTF ((stderr, "Stack %ld dies.\n", YY_CAST (long, yyk)));
              yymarkStackDeleted (yystackp, yyk);
              break;
            }
          else
            {
              YYRESULTTAG yyflag = yyglrReduce (yystackp, yyk, -yyaction,
                                                yyimmediate[-yyaction]);
              if (yyflag == yyerr)
                {
                  YY_DPRINTF ((stderr,
                               "Stack %ld dies "
                               "(predicate failure or explicit user error).\n",
                               YY_CAST (long, yyk)));
                  yymarkStackDeleted (yystackp, yyk);
                  break;
                }
              else if (yyflag != yyok)
                return yyflag;
            }
        }
    }
  return yyok;
}

static void
yyreportSyntaxError (yyGLRStack* yystackp)
{
  if (yystackp->yyerrState != 0)
    return;
#if ! YYERROR_VERBOSE
  yyerror (YY_("syntax error"));
#else
  {
  yySymbol yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);
  yybool yysize_overflow = yyfalse;
  char* yymsg = YY_NULLPTR;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat: reported tokens (one for the "unexpected",
     one per "expected"). */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Actual size of YYARG. */
  int yycount = 0;
  /* Cumulated lengths of YYARG.  */
  ptrdiff_t yysize = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[yystackp->yytops.yystates[0]->yylrState];
      ptrdiff_t yysize0 = yytnamerr (YY_NULLPTR, yytokenName (yytoken));
      yysize = yysize0;
      yyarg[yycount++] = yytokenName (yytoken);
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for this
             state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;
          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytokenName (yyx);
                {
                  ptrdiff_t yysz = yytnamerr (YY_NULLPTR, yytokenName (yyx));
                  if (YYSIZEMAX - yysize < yysz)
                    yysize_overflow = yytrue;
                  else
                    yysize += yysz;
                }
              }
        }
    }

  switch (yycount)
    {
#define YYCASE_(N, S)                   \
      case N:                           \
        yyformat = S;                   \
      break
    default: /* Avoid compiler warnings. */
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
#undef YYCASE_
    }

  {
    /* Don't count the "%s"s in the final size, but reserve room for
       the terminator.  */
    ptrdiff_t yysz = YY_CAST (ptrdiff_t, strlen (yyformat)) - 2 * yycount + 1;
    if (YYSIZEMAX - yysize < yysz)
      yysize_overflow = yytrue;
    else
      yysize += yysz;
  }

  if (!yysize_overflow)
    yymsg = YY_CAST (char *, YYMALLOC (YY_CAST (size_t, yysize)));

  if (yymsg)
    {
      char *yyp = yymsg;
      int yyi = 0;
      while ((*yyp = *yyformat))
        {
          if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
            {
              yyp += yytnamerr (yyp, yyarg[yyi++]);
              yyformat += 2;
            }
          else
            {
              ++yyp;
              ++yyformat;
            }
        }
      yyerror (yymsg);
      YYFREE (yymsg);
    }
  else
    {
      yyerror (YY_("syntax error"));
      yyMemoryExhausted (yystackp);
    }
  }
#endif /* YYERROR_VERBOSE */
  yynerrs += 1;
}

/* Recover from a syntax error on *YYSTACKP, assuming that *YYSTACKP->YYTOKENP,
   yylval, and yylloc are the syntactic category, semantic value, and location
   of the lookahead.  */
static void
yyrecoverSyntaxError (yyGLRStack* yystackp)
{
  if (yystackp->yyerrState == 3)
    /* We just shifted the error token and (perhaps) took some
       reductions.  Skip tokens until we can proceed.  */
    while (yytrue)
      {
        yySymbol yytoken;
        int yyj;
        if (yychar == YYEOF)
          yyFail (yystackp, YY_NULLPTR);
        if (yychar != YYEMPTY)
          {
            /* We throw away the lookahead, but the error range
               of the shifted error token must take it into account.  */
            yyGLRState *yys = yystackp->yytops.yystates[0];
            yyGLRStackItem yyerror_range[3];
            yyerror_range[1].yystate.yyloc = yys->yyloc;
            yyerror_range[2].yystate.yyloc = yylloc;
            YYLLOC_DEFAULT ((yys->yyloc), yyerror_range, 2);
            yytoken = YYTRANSLATE (yychar);
            yydestruct ("Error: discarding",
                        yytoken, &yylval, &yylloc);
            yychar = YYEMPTY;
          }
        yytoken = yygetToken (&yychar);
        yyj = yypact[yystackp->yytops.yystates[0]->yylrState];
        if (yypact_value_is_default (yyj))
          return;
        yyj += yytoken;
        if (yyj < 0 || YYLAST < yyj || yycheck[yyj] != yytoken)
          {
            if (yydefact[yystackp->yytops.yystates[0]->yylrState] != 0)
              return;
          }
        else if (! yytable_value_is_error (yytable[yyj]))
          return;
      }

  /* Reduce to one stack.  */
  {
    ptrdiff_t yyk;
    for (yyk = 0; yyk < yystackp->yytops.yysize; yyk += 1)
      if (yystackp->yytops.yystates[yyk] != YY_NULLPTR)
        break;
    if (yyk >= yystackp->yytops.yysize)
      yyFail (yystackp, YY_NULLPTR);
    for (yyk += 1; yyk < yystackp->yytops.yysize; yyk += 1)
      yymarkStackDeleted (yystackp, yyk);
    yyremoveDeletes (yystackp);
    yycompressStack (yystackp);
  }

  /* Now pop stack until we find a state that shifts the error token.  */
  yystackp->yyerrState = 3;
  while (yystackp->yytops.yystates[0] != YY_NULLPTR)
    {
      yyGLRState *yys = yystackp->yytops.yystates[0];
      int yyj = yypact[yys->yylrState];
      if (! yypact_value_is_default (yyj))
        {
          yyj += YYTERROR;
          if (0 <= yyj && yyj <= YYLAST && yycheck[yyj] == YYTERROR
              && yyisShiftAction (yytable[yyj]))
            {
              /* Shift the error token.  */
              int yyaction = yytable[yyj];
              /* First adjust its location.*/
              YYLTYPE yyerrloc;
              yystackp->yyerror_range[2].yystate.yyloc = yylloc;
              YYLLOC_DEFAULT (yyerrloc, (yystackp->yyerror_range), 2);
              YY_SYMBOL_PRINT ("Shifting", yystos[yyaction],
                               &yylval, &yyerrloc);
              yyglrShift (yystackp, 0, yyaction,
                          yys->yyposn, &yylval, &yyerrloc);
              yys = yystackp->yytops.yystates[0];
              break;
            }
        }
      yystackp->yyerror_range[1].yystate.yyloc = yys->yyloc;
      if (yys->yypred != YY_NULLPTR)
        yydestroyGLRState ("Error: popping", yys);
      yystackp->yytops.yystates[0] = yys->yypred;
      yystackp->yynextFree -= 1;
      yystackp->yyspaceLeft += 1;
    }
  if (yystackp->yytops.yystates[0] == YY_NULLPTR)
    yyFail (yystackp, YY_NULLPTR);
}

#define YYCHK1(YYE)                                                          \
  do {                                                                       \
    switch (YYE) {                                                           \
    case yyok:                                                               \
      break;                                                                 \
    case yyabort:                                                            \
      goto yyabortlab;                                                       \
    case yyaccept:                                                           \
      goto yyacceptlab;                                                      \
    case yyerr:                                                              \
      goto yyuser_error;                                                     \
    default:                                                                 \
      goto yybuglab;                                                         \
    }                                                                        \
  } while (0)

/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
  int yyresult;
  yyGLRStack yystack;
  yyGLRStack* const yystackp = &yystack;
  ptrdiff_t yyposn;

  YY_DPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY;
  yylval = yyval_default;
  yylloc = yyloc_default;

  if (! yyinitGLRStack (yystackp, YYINITDEPTH))
    goto yyexhaustedlab;
  switch (YYSETJMP (yystack.yyexception_buffer))
    {
    case 0: break;
    case 1: goto yyabortlab;
    case 2: goto yyexhaustedlab;
    default: goto yybuglab;
    }
  yyglrShift (&yystack, 0, 0, 0, &yylval, &yylloc);
  yyposn = 0;

  while (yytrue)
    {
      /* For efficiency, we have two loops, the first of which is
         specialized to deterministic operation (single stack, no
         potential ambiguity).  */
      /* Standard mode */
      while (yytrue)
        {
          yyStateNum yystate = yystack.yytops.yystates[0]->yylrState;
          YY_DPRINTF ((stderr, "Entering state %d\n", yystate));
          if (yystate == YYFINAL)
            goto yyacceptlab;
          if (yyisDefaultedState (yystate))
            {
              yyRuleNum yyrule = yydefaultAction (yystate);
              if (yyrule == 0)
                {
                  yystack.yyerror_range[1].yystate.yyloc = yylloc;
                  yyreportSyntaxError (&yystack);
                  goto yyuser_error;
                }
              YYCHK1 (yyglrReduce (&yystack, 0, yyrule, yytrue));
            }
          else
            {
              yySymbol yytoken = yygetToken (&yychar);
              const short* yyconflicts;
              int yyaction = yygetLRActions (yystate, yytoken, &yyconflicts);
              if (*yyconflicts != 0)
                break;
              if (yyisShiftAction (yyaction))
                {
                  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
                  yychar = YYEMPTY;
                  yyposn += 1;
                  yyglrShift (&yystack, 0, yyaction, yyposn, &yylval, &yylloc);
                  if (0 < yystack.yyerrState)
                    yystack.yyerrState -= 1;
                }
              else if (yyisErrorAction (yyaction))
                {
                  yystack.yyerror_range[1].yystate.yyloc = yylloc;                  yyreportSyntaxError (&yystack);
                  goto yyuser_error;
                }
              else
                YYCHK1 (yyglrReduce (&yystack, 0, -yyaction, yytrue));
            }
        }

      while (yytrue)
        {
          yySymbol yytoken_to_shift;
          ptrdiff_t yys;

          for (yys = 0; yys < yystack.yytops.yysize; yys += 1)
            yystackp->yytops.yylookaheadNeeds[yys] = yychar != YYEMPTY;

          /* yyprocessOneStack returns one of three things:

              - An error flag.  If the caller is yyprocessOneStack, it
                immediately returns as well.  When the caller is finally
                yyparse, it jumps to an error label via YYCHK1.

              - yyok, but yyprocessOneStack has invoked yymarkStackDeleted
                (&yystack, yys), which sets the top state of yys to NULL.  Thus,
                yyparse's following invocation of yyremoveDeletes will remove
                the stack.

              - yyok, when ready to shift a token.

             Except in the first case, yyparse will invoke yyremoveDeletes and
             then shift the next token onto all remaining stacks.  This
             synchronization of the shift (that is, after all preceding
             reductions on all stacks) helps prevent double destructor calls
             on yylval in the event of memory exhaustion.  */

          for (yys = 0; yys < yystack.yytops.yysize; yys += 1)
            YYCHK1 (yyprocessOneStack (&yystack, yys, yyposn));
          yyremoveDeletes (&yystack);
          if (yystack.yytops.yysize == 0)
            {
              yyundeleteLastStack (&yystack);
              if (yystack.yytops.yysize == 0)
                yyFail (&yystack, YY_("syntax error"));
              YYCHK1 (yyresolveStack (&yystack));
              YY_DPRINTF ((stderr, "Returning to deterministic operation.\n"));
              yystack.yyerror_range[1].yystate.yyloc = yylloc;
              yyreportSyntaxError (&yystack);
              goto yyuser_error;
            }

          /* If any yyglrShift call fails, it will fail after shifting.  Thus,
             a copy of yylval will already be on stack 0 in the event of a
             failure in the following loop.  Thus, yychar is set to YYEMPTY
             before the loop to make sure the user destructor for yylval isn't
             called twice.  */
          yytoken_to_shift = YYTRANSLATE (yychar);
          yychar = YYEMPTY;
          yyposn += 1;
          for (yys = 0; yys < yystack.yytops.yysize; yys += 1)
            {
              yyStateNum yystate = yystack.yytops.yystates[yys]->yylrState;
              const short* yyconflicts;
              int yyaction = yygetLRActions (yystate, yytoken_to_shift,
                              &yyconflicts);
              /* Note that yyconflicts were handled by yyprocessOneStack.  */
              YY_DPRINTF ((stderr, "On stack %ld, ", YY_CAST (long, yys)));
              YY_SYMBOL_PRINT ("shifting", yytoken_to_shift, &yylval, &yylloc);
              yyglrShift (&yystack, yys, yyaction, yyposn,
                          &yylval, &yylloc);
              YY_DPRINTF ((stderr, "Stack %ld now in state #%d\n",
                           YY_CAST (long, yys),
                           yystack.yytops.yystates[yys]->yylrState));
            }

          if (yystack.yytops.yysize == 1)
            {
              YYCHK1 (yyresolveStack (&yystack));
              YY_DPRINTF ((stderr, "Returning to deterministic operation.\n"));
              yycompressStack (&yystack);
              break;
            }
        }
      continue;
    yyuser_error:
      yyrecoverSyntaxError (&yystack);
      yyposn = yystack.yytops.yystates[0]->yyposn;
    }

 yyacceptlab:
  yyresult = 0;
  goto yyreturn;

 yybuglab:
  YY_ASSERT (yyfalse);
  goto yyabortlab;

 yyabortlab:
  yyresult = 1;
  goto yyreturn;

 yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturn;

 yyreturn:
  if (yychar != YYEMPTY)
    yydestruct ("Cleanup: discarding lookahead",
                YYTRANSLATE (yychar), &yylval, &yylloc);

  /* If the stack is well-formed, pop the stack until it is empty,
     destroying its entries as we go.  But free the stack regardless
     of whether it is well-formed.  */
  if (yystack.yyitems)
    {
      yyGLRState** yystates = yystack.yytops.yystates;
      if (yystates)
        {
          ptrdiff_t yysize = yystack.yytops.yysize;
          ptrdiff_t yyk;
          for (yyk = 0; yyk < yysize; yyk += 1)
            if (yystates[yyk])
              {
                while (yystates[yyk])
                  {
                    yyGLRState *yys = yystates[yyk];
                    yystack.yyerror_range[1].yystate.yyloc = yys->yyloc;
                    if (yys->yypred != YY_NULLPTR)
                      yydestroyGLRState ("Cleanup: popping", yys);
                    yystates[yyk] = yys->yypred;
                    yystack.yynextFree -= 1;
                    yystack.yyspaceLeft += 1;
                  }
                break;
              }
        }
      yyfreeGLRStack (&yystack);
    }

  return yyresult;
}

/* DEBUGGING ONLY */
#if YYDEBUG
static void
yy_yypstack (yyGLRState* yys)
{
  if (yys->yypred)
    {
      yy_yypstack (yys->yypred);
      YY_FPRINTF ((stderr, " -> "));
    }
  YY_FPRINTF ((stderr, "%d@%ld", yys->yylrState, YY_CAST (long, yys->yyposn)));
}

static void
yypstates (yyGLRState* yyst)
{
  if (yyst == YY_NULLPTR)
    YY_FPRINTF ((stderr, "<null>"));
  else
    yy_yypstack (yyst);
  YY_FPRINTF ((stderr, "\n"));
}

static void
yypstack (yyGLRStack* yystackp, ptrdiff_t yyk)
{
  yypstates (yystackp->yytops.yystates[yyk]);
}

static void
yypdumpstack (yyGLRStack* yystackp)
{
#define YYINDEX(YYX)                                                    \
  YY_CAST (long,                                                        \
           ((YYX)                                                       \
            ? YY_REINTERPRET_CAST (yyGLRStackItem*, (YYX)) - yystackp->yyitems \
            : -1))

  yyGLRStackItem* yyp;
  for (yyp = yystackp->yyitems; yyp < yystackp->yynextFree; yyp += 1)
    {
      YY_FPRINTF ((stderr, "%3ld. ",
                   YY_CAST (long, yyp - yystackp->yyitems)));
      if (*YY_REINTERPRET_CAST (yybool *, yyp))
        {
          YY_ASSERT (yyp->yystate.yyisState);
          YY_ASSERT (yyp->yyoption.yyisState);
          YY_FPRINTF ((stderr, "Res: %d, LR State: %d, posn: %ld, pred: %ld",
                       yyp->yystate.yyresolved, yyp->yystate.yylrState,
                       YY_CAST (long, yyp->yystate.yyposn),
                       YYINDEX (yyp->yystate.yypred)));
          if (! yyp->yystate.yyresolved)
            YY_FPRINTF ((stderr, ", firstVal: %ld",
                         YYINDEX (yyp->yystate.yysemantics.yyfirstVal)));
        }
      else
        {
          YY_ASSERT (!yyp->yystate.yyisState);
          YY_ASSERT (!yyp->yyoption.yyisState);
          YY_FPRINTF ((stderr, "Option. rule: %d, state: %ld, next: %ld",
                       yyp->yyoption.yyrule - 1,
                       YYINDEX (yyp->yyoption.yystate),
                       YYINDEX (yyp->yyoption.yynext)));
        }
      YY_FPRINTF ((stderr, "\n"));
    }

  YY_FPRINTF ((stderr, "Tops:"));
  {
    ptrdiff_t yyi;
    for (yyi = 0; yyi < yystackp->yytops.yysize; yyi += 1)
      YY_FPRINTF ((stderr, "%ld: %ld; ", YY_CAST (long, yyi),
                   YYINDEX (yystackp->yytops.yystates[yyi])));
    YY_FPRINTF ((stderr, "\n"));
  }
#undef YYINDEX
}
#endif

#undef yylval
#undef yychar
#undef yynerrs
#undef yylloc



#line 804 "bi/parser.ypp"

