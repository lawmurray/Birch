/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Skeleton implementation for Bison GLR parsers in C

   Copyright (C) 2002-2015 Free Software Foundation, Inc.

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

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.0.4"

/* Skeleton name.  */
#define YYSKELETON_NAME "glr.c"

/* Pure parsers.  */
#define YYPURE 0






/* First part of user declarations.  */

#line 55 "bi/parser.cpp" /* glr.c:240  */

# ifndef YY_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULLPTR nullptr
#  else
#   define YY_NULLPTR 0
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

/* Copy the second part of user declarations.  */

#line 88 "bi/parser.cpp" /* glr.c:263  */
/* Unqualified %code blocks.  */
#line 6 "bi/parser.ypp" /* glr.c:264  */

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

#line 155 "bi/parser.cpp" /* glr.c:264  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

#define YYSIZEMAX ((size_t) -1)

#ifdef __cplusplus
   typedef bool yybool;
#else
   typedef unsigned char yybool;
#endif
#define yytrue 1
#define yyfalse 0

#ifndef YYSETJMP
# include <setjmp.h>
# define YYJMP_BUF jmp_buf
# define YYSETJMP(Env) setjmp (Env)
/* Pacify clang.  */
# define YYLONGJMP(Env, Val) (longjmp (Env, Val), YYASSERT (0))
#endif

#ifndef YY_ATTRIBUTE
# if (defined __GNUC__                                               \
      && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__)))  \
     || defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#  define YY_ATTRIBUTE(Spec) __attribute__(Spec)
# else
#  define YY_ATTRIBUTE(Spec) /* empty */
# endif
#endif

#ifndef YY_ATTRIBUTE_PURE
# define YY_ATTRIBUTE_PURE   YY_ATTRIBUTE ((__pure__))
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# define YY_ATTRIBUTE_UNUSED YY_ATTRIBUTE ((__unused__))
#endif

#if !defined _Noreturn \
     && (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112)
# if defined _MSC_VER && 1200 <= _MSC_VER
#  define _Noreturn __declspec (noreturn)
# else
#  define _Noreturn YY_ATTRIBUTE ((__noreturn__))
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END \
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


#ifndef YYASSERT
# define YYASSERT(Condition) ((void) ((Condition) || (abort (), 0)))
#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  38
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   543

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  72
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  133
/* YYNRULES -- Number of rules.  */
#define YYNRULES  265
/* YYNRULES -- Number of states.  */
#define YYNSTATES  454
/* YYMAXRHS -- Maximum number of symbols on right-hand side of rule.  */
#define YYMAXRHS 10
/* YYMAXLEFT -- Maximum number of symbols to the left of a handle
   accessed by $0, $-1, etc., in any rule.  */
#define YYMAXLEFT 0

/* YYTRANSLATE(X) -- Bison symbol number corresponding to X.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   303

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    57,     2,     2,     2,     2,    71,     2,
      49,    50,    60,    58,    55,    59,    56,    61,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    64,    66,
      62,    67,    63,    53,    54,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    51,     2,    52,     2,    65,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    69,     2,    70,    68,     2,     2,     2,
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
static const unsigned short int yyrline[] =
{
       0,   160,   160,   169,   173,   177,   181,   185,   186,   187,
     188,   192,   196,   200,   204,   208,   212,   216,   220,   224,
     225,   226,   227,   228,   229,   230,   231,   235,   236,   240,
     241,   245,   249,   250,   251,   252,   253,   254,   255,   256,
     260,   261,   265,   266,   267,   271,   272,   276,   277,   281,
     282,   286,   287,   291,   292,   296,   297,   298,   299,   308,
     309,   313,   314,   318,   319,   323,   327,   328,   332,   336,
     337,   341,   345,   346,   350,   354,   355,   359,   360,   364,
     368,   369,   373,   377,   378,   382,   383,   387,   388,   392,
     396,   397,   401,   402,   406,   410,   411,   415,   416,   420,
     421,   425,   426,   430,   431,   435,   439,   440,   444,   445,
     449,   450,   454,   458,   459,   468,   469,   470,   471,   472,
     476,   477,   478,   479,   480,   484,   485,   486,   487,   488,
     489,   493,   493,   497,   497,   501,   502,   508,   508,   509,
     509,   513,   513,   514,   514,   518,   518,   522,   523,   524,
     525,   526,   527,   531,   535,   535,   539,   539,   543,   543,
     547,   547,   551,   552,   553,   557,   557,   558,   558,   559,
     559,   563,   564,   565,   569,   573,   577,   578,   579,   580,
     584,   588,   592,   593,   594,   598,   602,   606,   612,   613,
     617,   621,   625,   629,   633,   637,   638,   639,   640,   641,
     642,   643,   644,   645,   646,   647,   648,   649,   653,   654,
     658,   659,   663,   664,   665,   666,   667,   668,   672,   673,
     677,   678,   682,   683,   684,   685,   686,   687,   688,   689,
     690,   691,   695,   696,   700,   701,   705,   709,   713,   714,
     718,   722,   723,   727,   731,   732,   736,   740,   741,   745,
     754,   755,   759,   763,   764,   768,   769,   770,   771,   772,
     773,   774,   778,   779,   783,   784
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
  "PARALLEL", "DYNAMIC", "FINAL", "ABSTRACT", "NIL", "DOUBLE_BRACE_OPEN",
  "DOUBLE_BRACE_CLOSE", "NAME", "BOOL_LITERAL", "INT_LITERAL",
  "REAL_LITERAL", "STRING_LITERAL", "LEFT_OP", "RIGHT_OP", "LEFT_TILDE_OP",
  "RIGHT_TILDE_OP", "LEFT_QUERY_OP", "AND_OP", "OR_OP", "LE_OP", "GE_OP",
  "EQ_OP", "NE_OP", "RANGE_OP", "SPIN_OP", "'('", "')'", "'['", "']'",
  "'?'", "'@'", "','", "'.'", "'!'", "'+'", "'-'", "'*'", "'/'", "'<'",
  "'>'", "':'", "'_'", "';'", "'='", "'~'", "'{'", "'}'", "'&'", "$accept",
  "name", "bool_literal", "int_literal", "real_literal", "string_literal",
  "literal", "identifier", "parens_expression", "sequence_expression",
  "cast_expression", "function_expression", "this_expression",
  "super_expression", "nil_expression", "primary_expression",
  "index_expression", "index_list", "slice", "postfix_expression",
  "query_expression", "prefix_operator", "prefix_expression",
  "multiplicative_operator", "multiplicative_expression",
  "additive_operator", "additive_expression", "relational_operator",
  "relational_expression", "equality_operator", "equality_expression",
  "logical_and_operator", "logical_and_expression", "logical_or_operator",
  "logical_or_expression", "assign_operator", "assign_expression",
  "expression", "optional_expression", "expression_list",
  "span_expression", "span_list", "brackets", "parameters",
  "optional_parameters", "parameter_list", "parameter", "options",
  "option_list", "option", "arguments", "optional_arguments", "shape",
  "generics", "generic_list", "generic", "optional_generics",
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
  "parallel", "while", "do_while", "assertion", "return", "yield",
  "statement", "statements", "optional_statements", "class_statement",
  "class_statements", "optional_class_statements", "file_statement",
  "file_statements", "optional_file_statements", "file", "return_type",
  "optional_return_type", "value", "optional_value", "braces",
  "optional_braces", "class_braces", "optional_class_braces",
  "double_braces", "weak_modifier", "named_type", "primary_type", "type",
  "type_list", "optional_type_list", YY_NULLPTR
};
#endif

#define YYPACT_NINF -326
#define YYTABLE_NINF -236

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const short int yypact[] =
{
     142,     6,     6,     6,     6,    -2,    32,    32,  -326,  -326,
    -326,    -1,  -326,  -326,  -326,  -326,  -326,  -326,    66,  -326,
    -326,  -326,  -326,   512,  -326,  -326,    83,    37,   158,    40,
      40,    34,    65,  -326,  -326,    77,     6,  -326,  -326,    17,
    -326,     6,  -326,     6,    -3,  -326,    73,    73,  -326,  -326,
    -326,    79,  -326,   258,     6,  -326,    77,   110,   115,   145,
    -326,   -11,   141,  -326,   154,   156,   176,    55,   149,   166,
    -326,  -326,   171,   186,    18,   203,   203,    77,  -326,  -326,
    -326,  -326,  -326,  -326,  -326,  -326,  -326,  -326,  -326,  -326,
    -326,  -326,  -326,  -326,  -326,  -326,     6,   192,   159,   194,
      77,    -4,  -326,   178,     6,   476,   437,   132,  -326,    31,
    -326,    33,   181,   184,    20,    77,  -326,     6,  -326,   398,
    -326,  -326,  -326,  -326,  -326,     6,  -326,   206,   204,    77,
    -326,  -326,  -326,    87,   211,   203,   197,    77,  -326,  -326,
     213,  -326,   201,   212,    87,  -326,  -326,  -326,  -326,  -326,
     215,  -326,  -326,  -326,  -326,  -326,   476,   476,    73,   207,
    -326,  -326,  -326,  -326,  -326,  -326,  -326,  -326,  -326,  -326,
    -326,   218,  -326,  -326,   236,  -326,   476,  -326,    96,   126,
     -10,   153,   227,   152,  -326,  -326,  -326,   220,   228,   233,
    -326,   239,   231,   245,  -326,  -326,  -326,   240,  -326,  -326,
       6,  -326,   243,   151,  -326,     6,   476,     6,   476,   238,
     476,   476,   476,   296,  -326,   -38,    67,  -326,  -326,  -326,
    -326,  -326,  -326,   285,  -326,  -326,  -326,  -326,  -326,  -326,
     398,  -326,   241,  -326,  -326,  -326,     6,   125,    55,    55,
     203,  -326,  -326,   203,  -326,    77,     6,   260,   261,   203,
     259,     6,  -326,   476,  -326,     6,  -326,  -326,  -326,  -326,
     476,   476,   476,   476,   476,  -326,   476,   476,   476,  -326,
     197,   476,  -326,  -326,  -326,  -326,     6,    70,  -326,  -326,
     279,   238,  -326,   309,   238,   311,   257,  -326,   262,   264,
       6,    77,  -326,  -326,  -326,  -326,  -326,   476,   315,  -326,
    -326,  -326,  -326,  -326,  -326,    55,  -326,  -326,   269,  -326,
    -326,  -326,    55,   284,  -326,   280,   282,   289,  -326,  -326,
      96,   126,   -10,   153,   227,  -326,  -326,  -326,  -326,   271,
     290,  -326,   246,  -326,  -326,   272,   329,   476,  -326,   476,
    -326,  -326,  -326,   328,    61,   281,     6,    55,  -326,  -326,
    -326,   476,   476,  -326,   476,  -326,  -326,  -326,     6,     6,
     185,  -326,  -326,   286,  -326,   221,  -326,  -326,  -326,  -326,
    -326,   246,  -326,   275,  -326,    10,   304,   287,   476,  -326,
      45,   288,   291,  -326,   339,  -326,   305,  -326,  -326,    70,
      73,    73,     6,  -326,    77,     6,     6,  -326,  -326,  -326,
    -326,   476,  -326,   313,  -326,   297,  -326,  -326,   476,  -326,
    -326,   203,   203,  -326,    55,   160,    73,    73,   238,   476,
    -326,   318,  -326,  -326,    55,  -326,  -326,    64,   303,   306,
     203,   203,  -326,   238,   476,    55,    55,  -326,  -326,   307,
    -326,  -326,  -326,  -326,  -326,   238,  -326,  -326,  -326,    55,
      55,  -326,  -326,  -326
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const unsigned short int yydefact[] =
{
     164,     0,     0,     0,     0,     0,     0,     0,   162,   163,
       2,     0,   222,   223,   224,   225,   226,   227,     0,   228,
     229,   230,   231,   164,   234,   236,     0,     0,     0,   107,
     107,     0,     0,   174,   175,     0,     0,   233,     1,     0,
     145,     0,   173,     0,     0,   106,     0,     0,    44,    42,
      43,     0,   153,     0,     0,   249,     0,     0,   114,   253,
     255,     0,   107,    90,     0,     0,    92,     0,     0,     0,
     101,   105,     0,   103,     0,   239,   239,     0,    65,    68,
      57,    58,    61,    62,    51,    52,    47,    48,    55,    56,
     147,   148,   149,   150,   151,   152,     0,     0,   262,     0,
     265,     0,   113,   251,     0,     0,     0,     0,   256,   257,
     115,     0,     0,     0,    86,     0,    91,     0,   245,   211,
     244,   146,   171,   172,   102,     0,    83,     0,    87,     0,
     238,   131,   133,    89,     0,   239,     0,     0,   254,   264,
       0,   108,     0,   110,   112,   250,   252,   259,    16,    17,
       0,    18,     3,     4,     5,     6,     0,     0,     0,   114,
       7,     8,     9,    10,    19,    20,    21,    22,    23,    24,
      25,     0,    26,    32,    40,    45,     0,    49,    53,    59,
      63,    66,    69,    72,    74,   240,    95,    77,     0,    99,
      79,    80,     0,     0,   253,   258,   118,     0,   116,   117,
       0,    85,   167,   242,    93,     0,     0,     0,     0,     0,
       0,    76,     0,     0,   187,   114,     0,   196,   207,   198,
     197,   199,   200,     0,   201,   202,   203,   204,   205,   206,
     208,   210,     0,   195,   104,    84,     0,   237,     0,     0,
     239,   156,   263,   239,   109,     0,     0,     0,     0,   239,
      11,     0,    39,     0,    41,     0,    38,    36,    37,    46,
       0,     0,     0,     0,     0,    71,     0,     0,     0,    96,
       0,     0,    82,   260,   119,   169,     0,     0,   241,    94,
       0,     0,   185,     0,     0,     0,     0,    75,     0,     0,
       0,     0,   177,   178,   179,   181,   176,     0,     0,   209,
     243,    88,   132,   134,   154,     0,   261,   111,   114,    34,
      12,    13,     0,     0,    33,    29,     0,    28,    35,    50,
      54,    60,    64,    67,    70,    73,    78,   100,    81,     0,
      98,   248,   221,   247,   168,     0,   184,     0,   190,     0,
     192,   193,   194,     0,     0,     0,     0,     0,   157,    11,
      15,     0,     0,    31,     0,   170,    97,   165,     0,     0,
       0,   135,   136,     0,   212,     0,   213,   214,   215,   216,
     217,   218,   220,     0,   125,     0,     0,     0,     0,   126,
       0,     0,     0,   180,     0,   155,     0,    30,    27,     0,
       0,     0,     0,   160,     0,     0,     0,   219,   246,   183,
     182,     0,   191,     0,   129,     0,   127,   128,     0,    14,
     166,   239,   239,   158,     0,     0,     0,     0,     0,     0,
     130,     0,   139,   143,     0,   161,   120,     0,     0,     0,
     239,   239,   186,     0,     0,     0,     0,   159,   123,     0,
     121,   122,   137,   141,   189,     0,   140,   144,   124,     0,
       0,   188,   138,   142
};

  /* YYPGOTO[NTERM-NUM].  */
static const short int yypgoto[] =
{
    -326,     0,  -326,  -326,  -326,  -326,  -326,   -76,  -326,  -326,
    -326,  -326,  -326,  -326,  -326,  -326,  -326,    24,  -326,  -326,
    -326,   346,  -162,   317,   119,   330,   120,   331,   122,   333,
     123,   335,   124,   208,  -326,  -326,   130,    27,  -326,  -140,
    -326,   118,  -325,   -42,  -326,   157,   -25,  -326,   283,  -326,
    -101,  -326,   128,  -326,   274,  -326,     2,  -326,   161,  -326,
     -50,  -326,  -326,  -326,  -326,  -326,  -326,  -326,  -326,  -326,
    -326,  -326,  -326,  -326,  -326,  -326,  -326,  -326,  -326,  -326,
    -326,  -326,  -326,  -326,  -326,  -326,  -326,  -326,  -326,  -326,
    -326,  -326,  -326,  -107,  -314,  -326,  -326,  -326,    26,  -268,
    -326,  -326,  -326,  -326,  -326,  -326,  -326,  -326,  -326,   172,
    -326,  -326,    38,  -326,  -326,   380,  -326,  -326,    44,   -69,
    -192,  -326,   -89,  -154,  -326,    22,   411,  -326,   -28,   316,
     -26,   -79,  -326
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const short int yydefgoto[] =
{
      -1,   159,   160,   161,   162,   163,   164,   165,   166,   167,
     168,   169,   170,   171,   172,   173,   315,   316,   257,   174,
     175,   176,   177,   260,   178,   261,   179,   262,   180,   263,
     181,   264,   182,    95,   183,   267,   184,   187,   288,   188,
     191,   192,   111,    75,   202,   127,   128,    40,    65,    66,
     112,   357,   193,    45,    72,    73,    46,   102,   142,   143,
     250,    12,   364,   217,    13,   238,    14,   239,   365,   366,
     449,   435,   367,   450,   436,    15,    67,    96,    54,    16,
     347,    17,   305,   368,   424,   369,   414,    18,    19,   389,
     277,   329,    20,    21,    22,   297,   219,   220,   221,   283,
     222,   223,   224,   225,   226,   227,   228,   229,   230,   231,
     232,   371,   372,   373,    23,    24,    25,    26,   130,   131,
     113,   279,   120,   121,   333,   334,    33,   146,    59,    60,
      98,    99,   140
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const short int yytable[] =
{
      11,    27,    28,    29,    30,    76,    53,   132,   103,    61,
     197,   278,   218,    68,   259,    69,   247,   248,   370,   380,
     206,   139,   343,    11,   101,   105,   291,    10,    10,    97,
     233,    51,    47,    80,    81,    58,    62,    10,   106,    64,
     107,    58,   108,    58,    71,    56,   109,    31,    10,    10,
      57,   133,    88,    89,    51,   110,    58,   370,   242,   141,
      70,    32,    10,    35,   114,    10,   241,    63,   126,    74,
      36,   134,   201,   258,    51,   144,   147,    58,   384,   119,
      56,   194,   106,    38,   302,   303,    39,   200,   335,   203,
     427,    48,    49,    50,   106,    55,    51,   105,   319,   196,
      58,    58,    44,   237,    58,   292,   293,   294,    10,    58,
     106,   404,   107,   106,   108,    58,   249,    64,   109,   215,
     285,   118,    74,   218,   119,    71,    56,   379,   326,    58,
     438,    57,   185,   295,   190,   296,   331,    58,   136,   332,
     108,   233,  -235,    77,   109,     1,   216,     2,     3,     4,
       5,   348,   382,   148,   149,   150,    86,    87,   350,   100,
     151,     6,     7,    10,   152,   153,   154,   155,     8,     9,
     309,   304,   275,    10,   306,   314,   136,   101,   108,   318,
     312,   156,   109,   157,    84,    85,   158,   105,   265,    48,
      49,    50,   336,   385,    79,   338,   105,   189,    82,    83,
      58,   104,   136,    44,   108,   280,   116,   282,   109,   106,
     136,   107,   108,   108,   137,   122,   109,   109,   115,   144,
      41,   392,   129,   429,    42,    43,   426,   395,   396,   356,
     215,   117,   123,   281,   124,   284,    51,   286,   287,   289,
     129,   125,   135,   381,   138,    58,   308,   198,   330,   145,
     199,   308,   358,   359,   360,   308,   235,   216,   349,   236,
     425,   240,   189,   243,   244,   344,     7,   245,    78,   101,
     437,   246,   361,   362,   251,   268,    58,    10,   269,   405,
     317,   446,   447,   272,   252,   106,   400,   253,   270,   254,
     282,    58,   255,   256,   271,   452,   453,   273,   190,    78,
      79,    80,    81,    82,    83,   276,   274,   119,   290,   298,
     310,   300,   313,   311,   428,   105,    84,    85,    86,    87,
      88,    89,   337,   340,   345,   339,   439,   346,   341,   432,
     342,   101,   363,   351,   353,   352,   354,   355,   374,   106,
     375,   378,   422,   423,   444,   398,   282,   383,   411,   412,
     394,   401,   408,   402,   406,   409,   451,   407,   390,   391,
     419,   442,   443,   420,   376,   434,   377,   413,   415,   440,
      90,   363,   441,   448,   430,   431,   387,    52,   386,   317,
     320,   388,   321,    91,    92,   322,    93,   323,    94,   328,
     324,   266,    51,   301,    58,   416,   417,   325,   327,   234,
     204,   399,   299,    37,   393,   403,   307,   205,   206,   397,
     207,   410,   208,   209,   210,   211,   212,     6,    34,   148,
     149,   150,   213,   214,     0,   195,   151,     0,   418,    10,
     152,   153,   154,   155,     0,   421,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   433,   156,     0,   157,
       0,     0,   158,     0,     0,    48,    49,    50,   148,   149,
     150,   445,     0,     0,     0,   151,     0,   119,    10,   152,
     153,   154,   155,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   156,   186,   157,     0,
       0,   158,     0,     0,    48,    49,    50,   148,   149,   150,
       0,     0,     0,     0,   151,     0,     0,    10,   152,   153,
     154,   155,  -232,     0,     0,     1,     0,     2,     3,     4,
       5,     0,     0,     0,     0,   156,     0,   157,     0,     0,
     158,     6,     7,    48,    49,    50,     0,     0,     8,     9,
       0,     0,     0,    10
};

static const short int yycheck[] =
{
       0,     1,     2,     3,     4,    47,    31,    76,    58,    35,
     111,   203,   119,    41,   176,    43,   156,   157,   332,   344,
      10,   100,   290,    23,    62,    36,    64,    31,    31,    54,
     119,    31,    30,    43,    44,    35,    36,    31,    49,    39,
      51,    41,    53,    43,    44,    49,    57,    49,    31,    31,
      54,    77,    62,    63,    54,    66,    56,   371,   137,    63,
      63,    29,    31,    64,    62,    31,   135,    50,    50,    49,
       4,    96,   114,   174,    74,   101,   104,    77,   346,    69,
      49,   109,    49,     0,   238,   239,    49,    67,   280,   115,
     415,    57,    58,    59,    49,    30,    96,    36,   260,    66,
     100,   101,    62,   129,   104,    38,    39,    40,    31,   109,
      49,    66,    51,    49,    53,   115,   158,   117,    57,   119,
     209,    66,    49,   230,    69,   125,    49,    66,   268,   129,
      66,    54,   105,    66,   107,    68,    66,   137,    51,    69,
      53,   230,     0,    64,    57,     3,   119,     5,     6,     7,
       8,   305,   344,    21,    22,    23,    60,    61,   312,    49,
      28,    19,    20,    31,    32,    33,    34,    35,    26,    27,
     246,   240,   200,    31,   243,   251,    51,    62,    53,   255,
     249,    49,    57,    51,    58,    59,    54,    36,    36,    57,
      58,    59,   281,   347,    42,   284,    36,    65,    45,    46,
     200,    56,    51,    62,    53,   205,    50,   207,    57,    49,
      51,    51,    53,    53,    55,    66,    57,    57,    64,   245,
      62,    36,    37,   415,    66,    67,    66,     6,     7,   330,
     230,    55,    66,   206,    63,   208,   236,   210,   211,   212,
      37,    55,    50,   344,    50,   245,   246,    66,   276,    71,
      66,   251,     6,     7,     8,   255,    50,   230,   308,    55,
     414,    50,    65,    50,    63,   291,    20,    55,    41,    62,
     424,    56,    26,    27,    56,    55,   276,    31,    50,   380,
     253,   435,   436,    52,    48,    49,   375,    51,    55,    53,
     290,   291,    56,    57,    55,   449,   450,    52,   271,    41,
      42,    43,    44,    45,    46,    62,    66,    69,    12,    24,
      50,    70,    53,    52,   415,    36,    58,    59,    60,    61,
      62,    63,    13,    66,   297,    14,   427,    12,    66,   418,
      66,    62,   332,    49,    52,    55,    47,    66,    66,    49,
      11,    13,   411,   412,   433,    70,   346,    66,   390,   391,
      64,    47,    13,    66,    66,    50,   445,    66,   358,   359,
      47,   430,   431,    66,   337,    47,   339,   392,   394,    66,
      53,   371,    66,    66,   416,   417,   352,    31,   351,   352,
     261,   354,   262,    53,    53,   263,    53,   264,    53,   271,
     266,   183,   392,   236,   394,   395,   396,   267,   270,   125,
     117,   375,   230,    23,   360,   378,   245,     9,    10,   371,
      12,   389,    14,    15,    16,    17,    18,    19,     7,    21,
      22,    23,    24,    25,    -1,   109,    28,    -1,   401,    31,
      32,    33,    34,    35,    -1,   408,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   419,    49,    -1,    51,
      -1,    -1,    54,    -1,    -1,    57,    58,    59,    21,    22,
      23,   434,    -1,    -1,    -1,    28,    -1,    69,    31,    32,
      33,    34,    35,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    49,    50,    51,    -1,
      -1,    54,    -1,    -1,    57,    58,    59,    21,    22,    23,
      -1,    -1,    -1,    -1,    28,    -1,    -1,    31,    32,    33,
      34,    35,     0,    -1,    -1,     3,    -1,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    49,    -1,    51,    -1,    -1,
      54,    19,    20,    57,    58,    59,    -1,    -1,    26,    27,
      -1,    -1,    -1,    31
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,     3,     5,     6,     7,     8,    19,    20,    26,    27,
      31,    73,   133,   136,   138,   147,   151,   153,   159,   160,
     164,   165,   166,   186,   187,   188,   189,    73,    73,    73,
      73,    49,    29,   198,   198,    64,     4,   187,     0,    49,
     119,    62,    66,    67,    62,   125,   128,   128,    57,    58,
      59,    73,    93,   118,   150,    30,    49,    54,    73,   200,
     201,   202,    73,    50,    73,   120,   121,   148,   200,   200,
      63,    73,   126,   127,    49,   115,   115,    64,    41,    42,
      43,    44,    45,    46,    58,    59,    60,    61,    62,    63,
      95,    97,    99,   101,   103,   105,   149,   118,   202,   203,
      49,    62,   129,   132,    56,    36,    49,    51,    53,    57,
      66,   114,   122,   192,   128,    64,    50,    55,    66,    69,
     194,   195,    66,    66,    63,    55,    50,   117,   118,    37,
     190,   191,   191,   202,   118,    50,    51,    55,    50,   203,
     204,    63,   130,   131,   202,    71,   199,   200,    21,    22,
      23,    28,    32,    33,    34,    35,    49,    51,    54,    73,
      74,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    91,    92,    93,    94,    96,    98,
     100,   102,   104,   106,   108,   109,    50,   109,   111,    65,
     109,   112,   113,   124,   200,   201,    66,   122,    66,    66,
      67,   115,   116,   202,   120,     9,    10,    12,    14,    15,
      16,    17,    18,    24,    25,    73,   109,   135,   165,   168,
     169,   170,   172,   173,   174,   175,   176,   177,   178,   179,
     180,   181,   182,   194,   126,    50,    55,   202,   137,   139,
      50,   191,   203,    50,    63,    55,    56,   111,   111,   115,
     132,    56,    48,    51,    53,    56,    57,    90,   122,    94,
      95,    97,    99,   101,   103,    36,   105,   107,    55,    50,
      55,    55,    52,    52,    66,   200,    62,   162,   192,   193,
      73,   109,    73,   171,   109,   194,   109,   109,   110,   109,
      12,    64,    38,    39,    40,    66,    68,   167,    24,   181,
      70,   117,   195,   195,   191,   154,   191,   130,    73,    79,
      50,    52,   191,    53,    79,    88,    89,   109,    79,    94,
      96,    98,   100,   102,   104,   108,   111,   124,   113,   163,
     200,    66,    69,   196,   197,   192,   194,    13,   194,    14,
      66,    66,    66,   171,   202,   109,    12,   152,   195,   132,
     195,    49,    55,    52,    47,    66,   122,   123,     6,     7,
       8,    26,    27,    73,   134,   140,   141,   144,   155,   157,
     166,   183,   184,   185,    66,    11,   109,   109,    13,    66,
     114,   122,   192,    66,   171,   195,   109,    89,   109,   161,
      73,    73,    36,   190,    64,     6,     7,   184,    70,   170,
     194,    47,    66,   109,    66,   122,    66,    66,    13,    50,
     197,   115,   115,   118,   158,   202,    73,    73,   109,    47,
      66,   109,   191,   191,   156,   195,    66,   114,   122,   192,
     115,   115,   194,   109,    47,   143,   146,   195,    66,   122,
      66,    66,   191,   191,   194,   109,   195,   195,    66,   142,
     145,   194,   195,   195
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,    72,    73,    74,    75,    76,    77,    78,    78,    78,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      87,    87,    87,    87,    87,    87,    87,    88,    88,    89,
      89,    90,    91,    91,    91,    91,    91,    91,    91,    91,
      92,    92,    93,    93,    93,    94,    94,    95,    95,    96,
      96,    97,    97,    98,    98,    99,    99,    99,    99,   100,
     100,   101,   101,   102,   102,   103,   104,   104,   105,   106,
     106,   107,   108,   108,   109,   110,   110,   111,   111,   112,
     113,   113,   114,   115,   115,   116,   116,   117,   117,   118,
     119,   119,   120,   120,   121,   122,   122,   123,   123,   124,
     124,   125,   125,   126,   126,   127,   128,   128,   129,   129,
     130,   130,   131,   132,   132,   133,   133,   133,   133,   133,
     134,   134,   134,   134,   134,   135,   135,   135,   135,   135,
     135,   137,   136,   139,   138,   140,   140,   142,   141,   143,
     141,   145,   144,   146,   144,   148,   147,   149,   149,   149,
     149,   149,   149,   150,   152,   151,   154,   153,   156,   155,
     158,   157,   159,   159,   159,   161,   160,   162,   160,   163,
     160,   164,   164,   164,   165,   166,   167,   167,   167,   167,
     168,   169,   170,   170,   170,   171,   172,   173,   174,   174,
     175,   176,   177,   178,   179,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   181,   181,
     182,   182,   183,   183,   183,   183,   183,   183,   184,   184,
     185,   185,   186,   186,   186,   186,   186,   186,   186,   186,
     186,   186,   187,   187,   188,   188,   189,   190,   191,   191,
     192,   193,   193,   194,   195,   195,   196,   197,   197,   198,
     199,   199,   200,   201,   201,   202,   202,   202,   202,   202,
     202,   202,   203,   203,   204,   204
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const unsigned char yyr2[] =
{
       0,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     2,     3,     3,     6,     4,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     3,     1,     1,
       3,     3,     1,     3,     3,     3,     2,     2,     2,     2,
       1,     2,     1,     1,     1,     1,     2,     1,     1,     1,
       3,     1,     1,     1,     3,     1,     1,     1,     1,     1,
       3,     1,     1,     1,     3,     1,     1,     3,     1,     1,
       3,     1,     1,     3,     1,     1,     0,     1,     3,     1,
       1,     3,     3,     2,     3,     1,     0,     1,     3,     3,
       2,     3,     1,     3,     4,     2,     3,     1,     0,     1,
       3,     2,     3,     1,     3,     1,     1,     0,     2,     3,
       1,     3,     1,     1,     0,     4,     5,     5,     5,     6,
       4,     5,     5,     5,     6,     4,     4,     5,     5,     5,
       6,     0,     7,     0,     7,     1,     1,     0,     7,     0,
       6,     0,     7,     0,     6,     0,     5,     1,     1,     1,
       1,     1,     1,     1,     0,     9,     0,     8,     0,     5,
       0,     4,     1,     1,     0,     0,    10,     0,     7,     0,
       8,     5,     5,     3,     2,     2,     1,     1,     1,     1,
       4,     2,     5,     5,     3,     1,     7,     1,     9,     8,
       3,     5,     3,     3,     3,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       1,     0,     1,     1,     1,     1,     1,     1,     1,     2,
       1,     0,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     2,     1,     0,     1,     2,     1,     0,
       2,     1,     0,     3,     1,     1,     3,     1,     1,     2,
       1,     0,     3,     1,     3,     1,     2,     2,     3,     3,
       4,     5,     1,     3,     1,     0
};


/* YYDPREC[RULE-NUM] -- Dynamic precedence of rule #RULE-NUM (0 if none).  */
static const unsigned char yydprec[] =
{
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     3,
       2,     0,     0,     0,     0,     0,     0,     0,     0,     0,
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
static const unsigned char yymerger[] =
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
       0,     0,     0,     0,     0,     0
};

/* YYCONFLP[YYPACT[STATE-NUM]] -- Pointer into YYCONFL of start of
   list of conflicting reductions corresponding to action entry for
   state STATE-NUM in yytable.  0 means no conflicts.  The list in
   yyconfl is terminated by a rule number of 0.  */
static const unsigned char yyconflp[] =
{
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     9,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       5,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    11,     1,    13,     0,
       0,     0,    15,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     3,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     7,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    17,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    19,     0,     0,     0,     0,     0,     0,     0,     0,
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
       0,     0,     0,     0
};

/* YYCONFL[I] -- lists of conflicting rule numbers, each terminated by
   0, pointed into by YYCONFLP.  */
static const short int yyconfl[] =
{
       0,   114,     0,   107,     0,   257,     0,   114,     0,   114,
       0,   237,     0,   237,     0,   237,     0,    11,     0,   114,
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


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL

/* Print *YYLOCP on YYO.  Private, do not rely on its existence. */

YY_ATTRIBUTE_UNUSED
static unsigned
yy_location_print_ (FILE *yyo, YYLTYPE const * const yylocp)
{
  unsigned res = 0;
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


# define YYDPRINTF(Args)                        \
  do {                                          \
    if (yydebug)                                \
      YYFPRINTF Args;                           \
  } while (0)


/*----------------------------------------.
| Print this symbol's value on YYOUTPUT.  |
`----------------------------------------*/

static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
{
  FILE *yyo = yyoutput;
  YYUSE (yyo);
  YYUSE (yylocationp);
  if (!yyvaluep)
    return;
  YYUSE (yytype);
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
{
  YYFPRINTF (yyoutput, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  YY_LOCATION_PRINT (yyoutput, *yylocationp);
  YYFPRINTF (yyoutput, ": ");
  yy_symbol_value_print (yyoutput, yytype, yyvaluep, yylocationp);
  YYFPRINTF (yyoutput, ")");
}

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                  \
  do {                                                                  \
    if (yydebug)                                                        \
      {                                                                 \
        YYFPRINTF (stderr, "%s ", Title);                               \
        yy_symbol_print (stderr, Type, Value, Location);        \
        YYFPRINTF (stderr, "\n");                                       \
      }                                                                 \
  } while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;

struct yyGLRStack;
static void yypstack (struct yyGLRStack* yystackp, size_t yyk)
  YY_ATTRIBUTE_UNUSED;
static void yypdumpstack (struct yyGLRStack* yystackp)
  YY_ATTRIBUTE_UNUSED;

#else /* !YYDEBUG */

# define YYDPRINTF(Args)
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
static size_t
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      size_t yyn = 0;
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
            /* Fall through.  */
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

  if (! yyres)
    return strlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

#endif /* !YYERROR_VERBOSE */

/** State numbers, as in LALR(1) machine */
typedef int yyStateNum;

/** Rule numbers, as in LALR(1) machine */
typedef int yyRuleNum;

/** Grammar symbol */
typedef int yySymbol;

/** Item references, as in LALR(1) machine */
typedef short int yyItemNum;

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
  size_t yyposn;
  union {
    /** First in a chain of alternative reductions producing the
     *  non-terminal corresponding to this state, threaded through
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
  size_t yysize, yycapacity;
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
  size_t yyspaceLeft;
  yyGLRState* yysplitPoint;
  yyGLRState* yylastDeleted;
  yyGLRStateSet yytops;
};

#if YYSTACKEXPANDABLE
static void yyexpandGLRStack (yyGLRStack* yystackp);
#endif

static _Noreturn void
yyFail (yyGLRStack* yystackp, const char* yymsg)
{
  if (yymsg != YY_NULLPTR)
    yyerror (yymsg);
  YYLONGJMP (yystackp->yyexception_buffer, 1);
}

static _Noreturn void
yyMemoryExhausted (yyGLRStack* yystackp)
{
  YYLONGJMP (yystackp->yyexception_buffer, 2);
}

#if YYDEBUG || YYERROR_VERBOSE
/** A printable representation of TOKEN.  */
static inline const char*
yytokenName (yySymbol yytoken)
{
  if (yytoken == YYEMPTY)
    return "";

  return yytname[yytoken];
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
yyuserAction (yyRuleNum yyn, size_t yyrhslen, yyGLRStackItem* yyvsp,
              yyGLRStack* yystackp,
              YYSTYPE* yyvalp, YYLTYPE *yylocp)
{
  yybool yynormal YY_ATTRIBUTE_UNUSED = (yystackp->yysplitPoint == YY_NULLPTR);
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
# define YYFILL(N) yyfill (yyvsp, &yylow, N, yynormal)
# undef YYBACKUP
# define YYBACKUP(Token, Value)                                              \
  return yyerror (YY_("syntax error: cannot back up")),     \
         yyerrok, yyerr

  yylow = 1;
  if (yyrhslen == 0)
    *yyvalp = yyval_default;
  else
    *yyvalp = yyvsp[YYFILL (1-yyrhslen)].yystate.yysemantics.yysval;
  YYLLOC_DEFAULT ((*yylocp), (yyvsp - yyrhslen), yyrhslen);
  yystackp->yyerror_range[1].yystate.yyloc = *yylocp;

  switch (yyn)
    {
        case 2:
#line 160 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valString)); }
#line 1451 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 3:
#line 169 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Literal<bool>((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valString), make_loc((*yylocp))); }
#line 1457 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 4:
#line 173 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Literal<int64_t>((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valString), make_loc((*yylocp))); }
#line 1463 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 5:
#line 177 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Literal<double>((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valString), make_loc((*yylocp))); }
#line 1469 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 6:
#line 181 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Literal<const char*>((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valString), make_loc((*yylocp))); }
#line 1475 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 11:
#line 192 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::NamedExpression((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 1481 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 12:
#line 196 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Parentheses((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1487 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 13:
#line 200 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Sequence((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1493 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 14:
#line 204 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Cast(new bi::NamedType(false, (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1499 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 15:
#line 208 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LambdaFunction((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 1505 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 16:
#line 212 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::This(make_loc((*yylocp))); }
#line 1511 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 17:
#line 216 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Super(make_loc((*yylocp))); }
#line 1517 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 18:
#line 220 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Nil(make_loc((*yylocp))); }
#line 1523 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 27:
#line 235 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Range((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1529 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 28:
#line 236 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Index((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1535 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 30:
#line 241 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1541 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 31:
#line 245 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1547 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 33:
#line 250 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Member((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1553 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 34:
#line 251 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Global((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1559 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 35:
#line 252 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Member((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1565 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 36:
#line 253 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Slice((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1571 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 37:
#line 254 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Call((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1577 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 38:
#line 255 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Get((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1583 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 39:
#line 256 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Spin((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1589 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 41:
#line 261 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Query((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1595 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 42:
#line 265 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("+"); }
#line 1601 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 43:
#line 266 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("-"); }
#line 1607 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 44:
#line 267 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("!"); }
#line 1613 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 46:
#line 272 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::UnaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1619 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 47:
#line 276 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("*"); }
#line 1625 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 48:
#line 277 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("/"); }
#line 1631 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 50:
#line 282 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1637 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 51:
#line 286 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("+"); }
#line 1643 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 52:
#line 287 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("-"); }
#line 1649 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 54:
#line 292 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1655 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 55:
#line 296 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("<"); }
#line 1661 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 56:
#line 297 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name(">"); }
#line 1667 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 57:
#line 298 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("<="); }
#line 1673 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 58:
#line 299 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name(">="); }
#line 1679 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 60:
#line 309 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1685 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 61:
#line 313 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("=="); }
#line 1691 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 62:
#line 314 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("!="); }
#line 1697 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 64:
#line 319 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1703 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 65:
#line 323 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("&&"); }
#line 1709 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 67:
#line 328 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1715 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 68:
#line 332 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("||"); }
#line 1721 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 70:
#line 337 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1727 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 71:
#line 341 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("<-"); }
#line 1733 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 73:
#line 346 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Assign((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1739 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 76:
#line 355 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1745 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 78:
#line 360 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1751 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 79:
#line 364 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Span((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1757 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 81:
#line 369 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1763 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 82:
#line 373 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1769 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 83:
#line 377 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1775 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 84:
#line 378 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1781 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 86:
#line 383 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1787 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 88:
#line 388 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1793 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 89:
#line 392 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Parameter(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1799 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 90:
#line 396 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1805 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 91:
#line 397 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1811 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 93:
#line 402 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1817 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 94:
#line 406 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Parameter(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1823 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 95:
#line 410 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1829 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 96:
#line 411 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1835 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 98:
#line 416 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1841 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 99:
#line 420 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valInt) = 1; }
#line 1847 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 100:
#line 421 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valInt) = (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valInt) + 1; }
#line 1853 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 101:
#line 425 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1859 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 102:
#line 426 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1865 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 104:
#line 431 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1871 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 105:
#line 435 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Generic(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valName), empty_type((*yylocp)), make_loc((*yylocp))); }
#line 1877 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 107:
#line 440 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1883 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 108:
#line 444 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 1889 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 109:
#line 445 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType); }
#line 1895 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 111:
#line 450 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::TypeList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 1901 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 114:
#line 459 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 1907 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 115:
#line 468 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1913 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 116:
#line 469 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1919 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 117:
#line 470 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 1925 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 118:
#line 471 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), new bi::ArrayType((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression)->width(), make_loc((*yylocp))), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1931 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 119:
#line 472 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), new bi::ArrayType((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression)->width(), make_loc((*yylocp))), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1937 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 120:
#line 476 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1943 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 121:
#line 477 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1949 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 122:
#line 478 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 1955 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 123:
#line 479 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), new bi::ArrayType((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression)->width(), make_loc((*yylocp))), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1961 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 124:
#line 480 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), new bi::ArrayType((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression)->width(), make_loc((*yylocp))), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1967 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 125:
#line 484 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::LocalVariable(bi::AUTO, (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), empty_type((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 1973 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 126:
#line 485 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::LocalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1979 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 127:
#line 486 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::LocalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1985 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 128:
#line 487 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::LocalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 1991 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 129:
#line 488 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::LocalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), new bi::ArrayType((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression)->width(), make_loc((*yylocp))), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1997 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 130:
#line 489 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::LocalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), new bi::ArrayType((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression)->width(), make_loc((*yylocp))), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2003 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 131:
#line 493 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2009 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 132:
#line 493 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Function(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2015 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 133:
#line 497 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2021 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 134:
#line 497 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Fiber(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2027 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 135:
#line 501 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valAnnotation) = bi::FINAL; }
#line 2033 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 136:
#line 502 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valAnnotation) = bi::ABSTRACT; }
#line 2039 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 137:
#line 508 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2045 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 138:
#line 508 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::MemberFunction((((yyGLRStackItem const *)yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valAnnotation), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2051 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 139:
#line 509 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2057 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 140:
#line 509 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::MemberFunction(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2063 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 141:
#line 513 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2069 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 142:
#line 513 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::MemberFiber((((yyGLRStackItem const *)yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valAnnotation), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2075 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 143:
#line 514 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2081 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 144:
#line 514 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::MemberFiber(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2087 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 145:
#line 518 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2093 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 146:
#line 518 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Program((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2099 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 154:
#line 535 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2105 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 155:
#line 535 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::BinaryOperator(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2111 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 156:
#line 539 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2117 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 157:
#line 539 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::UnaryOperator(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2123 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 158:
#line 543 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2129 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 159:
#line 543 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::AssignmentOperator((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2135 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 160:
#line 547 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2141 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 161:
#line 547 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::ConversionOperator((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2147 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 162:
#line 551 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valAnnotation) = bi::FINAL; }
#line 2153 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 163:
#line 552 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valAnnotation) = bi::ABSTRACT; }
#line 2159 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 164:
#line 553 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valAnnotation) = bi::NONE; }
#line 2165 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 165:
#line 557 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2171 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 166:
#line 557 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Class((((yyGLRStackItem const *)yyvsp)[YYFILL (-9)].yystate.yysemantics.yysval.valAnnotation), (((yyGLRStackItem const *)yyvsp)[YYFILL (-7)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), false, (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2177 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 167:
#line 558 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2183 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 168:
#line 558 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Class((((yyGLRStackItem const *)yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valAnnotation), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), empty_type((*yylocp)), false, empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2189 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 169:
#line 559 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2195 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 170:
#line 559 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Class((((yyGLRStackItem const *)yyvsp)[YYFILL (-7)].yystate.yysemantics.yysval.valAnnotation), (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), true, empty_expr((*yylocp)), empty_stmt((*yylocp)), make_doc_loc((*yylocp))); }
#line 2201 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 171:
#line 563 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Basic(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), false, make_doc_loc((*yylocp))); }
#line 2207 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 172:
#line 564 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Basic(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), true, make_doc_loc((*yylocp))); }
#line 2213 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 173:
#line 565 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Basic(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), empty_expr((*yylocp)), empty_type((*yylocp)), false, make_doc_loc((*yylocp))); }
#line 2219 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 174:
#line 569 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Raw(new bi::Name("cpp"), pop_raw(), make_loc((*yylocp))); }
#line 2225 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 175:
#line 573 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Raw(new bi::Name("hpp"), pop_raw(), make_loc((*yylocp))); }
#line 2231 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 176:
#line 577 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("~"); }
#line 2237 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 177:
#line 578 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("<~"); }
#line 2243 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 178:
#line 579 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("~>"); }
#line 2249 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 179:
#line 580 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("<-?"); }
#line 2255 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 180:
#line 584 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Assume((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2261 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 181:
#line 588 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::ExpressionStatement((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2267 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 182:
#line 592 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::If((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2273 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 183:
#line 593 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::If((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2279 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 184:
#line 594 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::If((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), empty_stmt((*yylocp)), make_loc((*yylocp))); }
#line 2285 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 185:
#line 598 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::LocalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valName), new bi::NamedType(new bi::Name("Integer")), make_loc((*yylocp))); }
#line 2291 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 186:
#line 602 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::For(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2297 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 187:
#line 606 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valAnnotation) = bi::DYNAMIC; }
#line 2303 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 188:
#line 612 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Parallel((((yyGLRStackItem const *)yyvsp)[YYFILL (-8)].yystate.yysemantics.yysval.valAnnotation), (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2309 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 189:
#line 613 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Parallel(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2315 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 190:
#line 617 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::While((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2321 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 191:
#line 621 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::DoWhile((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2327 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 192:
#line 625 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Assert((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2333 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 193:
#line 629 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Return((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2339 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 194:
#line 633 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Yield((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2345 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 209:
#line 654 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::StatementList((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2351 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 211:
#line 659 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2357 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 219:
#line 673 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::StatementList((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2363 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 221:
#line 678 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2369 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 233:
#line 696 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::StatementList((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2375 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 235:
#line 701 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2381 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 236:
#line 705 "bi/parser.ypp" /* glr.c:816  */
    { compiler->setRoot((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement)); }
#line 2387 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 237:
#line 709 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType); }
#line 2393 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 239:
#line 714 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 2399 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 240:
#line 718 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression); }
#line 2405 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 242:
#line 723 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 2411 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 243:
#line 727 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Braces((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2417 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 245:
#line 732 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2423 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 246:
#line 736 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Braces((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2429 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 248:
#line 741 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2435 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 250:
#line 754 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valBool) = true; }
#line 2441 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 251:
#line 755 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valBool) = false; }
#line 2447 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 252:
#line 759 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::NamedType((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valBool), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2453 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 254:
#line 764 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::TupleType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2459 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 256:
#line 769 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::OptionalType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2465 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 257:
#line 770 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::FiberType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_type((*yylocp)), make_loc((*yylocp))); }
#line 2471 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 258:
#line 771 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::FiberType((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2477 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 259:
#line 772 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::MemberType((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2483 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 260:
#line 773 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::ArrayType((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valInt), make_loc((*yylocp))); }
#line 2489 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 261:
#line 774 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::FunctionType((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2495 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 263:
#line 779 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::TypeList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2501 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 265:
#line 784 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 2507 "bi/parser.cpp" /* glr.c:816  */
    break;


#line 2511 "bi/parser.cpp" /* glr.c:816  */
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
            YYFPRINTF (stderr, "%s unresolved", yymsg);
          else
            YYFPRINTF (stderr, "%s incomplete", yymsg);
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

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-326)))

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

#define yytable_value_is_error(Yytable_value) \
  0

/** Set *YYACTION to the action to take in YYSTATE on seeing YYTOKEN.
 *  Result R means
 *    R < 0:  Reduce on rule -R.
 *    R = 0:  Error.
 *    R > 0:  Shift to state R.
 *  Set *YYCONFLICTS to a pointer into yyconfl to a 0-terminated list
 *  of conflicting reductions.
 */
static inline void
yygetLRActions (yyStateNum yystate, int yytoken,
                int* yyaction, const short int** yyconflicts)
{
  int yyindex = yypact[yystate] + yytoken;
  if (yypact_value_is_default (yypact[yystate])
      || yyindex < 0 || YYLAST < yyindex || yycheck[yyindex] != yytoken)
    {
      *yyaction = -yydefact[yystate];
      *yyconflicts = yyconfl;
    }
  else if (! yytable_value_is_error (yytable[yyindex]))
    {
      *yyaction = yytable[yyindex];
      *yyconflicts = yyconfl + yyconflp[yyindex];
    }
  else
    {
      *yyaction = 0;
      *yyconflicts = yyconfl + yyconflp[yyindex];
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
yyaddDeferredAction (yyGLRStack* yystackp, size_t yyk, yyGLRState* yystate,
                     yyGLRState* yyrhs, yyRuleNum yyrule)
{
  yySemanticOption* yynewOption =
    &yynewGLRStackItem (yystackp, yyfalse)->yyoption;
  YYASSERT (!yynewOption->yyisState);
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
  yyset->yystates = (yyGLRState**) YYMALLOC (16 * sizeof yyset->yystates[0]);
  if (! yyset->yystates)
    return yyfalse;
  yyset->yystates[0] = YY_NULLPTR;
  yyset->yylookaheadNeeds =
    (yybool*) YYMALLOC (16 * sizeof yyset->yylookaheadNeeds[0]);
  if (! yyset->yylookaheadNeeds)
    {
      YYFREE (yyset->yystates);
      return yyfalse;
    }
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
yyinitGLRStack (yyGLRStack* yystackp, size_t yysize)
{
  yystackp->yyerrState = 0;
  yynerrs = 0;
  yystackp->yyspaceLeft = yysize;
  yystackp->yyitems =
    (yyGLRStackItem*) YYMALLOC (yysize * sizeof yystackp->yynextFree[0]);
  if (!yystackp->yyitems)
    return yyfalse;
  yystackp->yynextFree = yystackp->yyitems;
  yystackp->yysplitPoint = YY_NULLPTR;
  yystackp->yylastDeleted = YY_NULLPTR;
  return yyinitStateSet (&yystackp->yytops);
}


#if YYSTACKEXPANDABLE
# define YYRELOC(YYFROMITEMS,YYTOITEMS,YYX,YYTYPE) \
  &((YYTOITEMS) - ((YYFROMITEMS) - (yyGLRStackItem*) (YYX)))->YYTYPE

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
  size_t yynewSize;
  size_t yyn;
  size_t yysize = yystackp->yynextFree - yystackp->yyitems;
  if (YYMAXDEPTH - YYHEADROOM < yysize)
    yyMemoryExhausted (yystackp);
  yynewSize = 2*yysize;
  if (YYMAXDEPTH < yynewSize)
    yynewSize = YYMAXDEPTH;
  yynewItems = (yyGLRStackItem*) YYMALLOC (yynewSize * sizeof yynewItems[0]);
  if (! yynewItems)
    yyMemoryExhausted (yystackp);
  for (yyp0 = yystackp->yyitems, yyp1 = yynewItems, yyn = yysize;
       0 < yyn;
       yyn -= 1, yyp0 += 1, yyp1 += 1)
    {
      *yyp1 = *yyp0;
      if (*(yybool *) yyp0)
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
yymarkStackDeleted (yyGLRStack* yystackp, size_t yyk)
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
  YYDPRINTF ((stderr, "Restoring last deleted stack as stack #0.\n"));
  yystackp->yylastDeleted = YY_NULLPTR;
}

static inline void
yyremoveDeletes (yyGLRStack* yystackp)
{
  size_t yyi, yyj;
  yyi = yyj = 0;
  while (yyj < yystackp->yytops.yysize)
    {
      if (yystackp->yytops.yystates[yyi] == YY_NULLPTR)
        {
          if (yyi == yyj)
            {
              YYDPRINTF ((stderr, "Removing dead stacks.\n"));
            }
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
            {
              YYDPRINTF ((stderr, "Rename stack %lu -> %lu.\n",
                          (unsigned long int) yyi, (unsigned long int) yyj));
            }
          yyj += 1;
        }
      yyi += 1;
    }
}

/** Shift to a new state on stack #YYK of *YYSTACKP, corresponding to LR
 * state YYLRSTATE, at input position YYPOSN, with (resolved) semantic
 * value *YYVALP and source location *YYLOCP.  */
static inline void
yyglrShift (yyGLRStack* yystackp, size_t yyk, yyStateNum yylrState,
            size_t yyposn,
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
yyglrShiftDefer (yyGLRStack* yystackp, size_t yyk, yyStateNum yylrState,
                 size_t yyposn, yyGLRState* yyrhs, yyRuleNum yyrule)
{
  yyGLRState* yynewState = &yynewGLRStackItem (yystackp, yytrue)->yystate;
  YYASSERT (yynewState->yyisState);

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
do {                                    \
  if (yydebug)                          \
    yy_reduce_print Args;               \
} while (0)

/*----------------------------------------------------------------------.
| Report that stack #YYK of *YYSTACKP is going to be reduced by YYRULE. |
`----------------------------------------------------------------------*/

static inline void
yy_reduce_print (int yynormal, yyGLRStackItem* yyvsp, size_t yyk,
                 yyRuleNum yyrule)
{
  int yynrhs = yyrhsLength (yyrule);
  int yylow = 1;
  int yyi;
  YYFPRINTF (stderr, "Reducing stack %lu by rule %d (line %lu):\n",
             (unsigned long int) yyk, yyrule - 1,
             (unsigned long int) yyrline[yyrule]);
  if (! yynormal)
    yyfillin (yyvsp, 1, -yynrhs);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[yyvsp[yyi - yynrhs + 1].yystate.yylrState],
                       &yyvsp[yyi - yynrhs + 1].yystate.yysemantics.yysval
                       , &(((yyGLRStackItem const *)yyvsp)[YYFILL ((yyi + 1) - (yynrhs))].yystate.yyloc)                       );
      if (!yyvsp[yyi - yynrhs + 1].yystate.yyresolved)
        YYFPRINTF (stderr, " (unresolved)");
      YYFPRINTF (stderr, "\n");
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
yydoAction (yyGLRStack* yystackp, size_t yyk, yyRuleNum yyrule,
            YYSTYPE* yyvalp, YYLTYPE *yylocp)
{
  int yynrhs = yyrhsLength (yyrule);

  if (yystackp->yysplitPoint == YY_NULLPTR)
    {
      /* Standard special case: single stack.  */
      yyGLRStackItem* yyrhs = (yyGLRStackItem*) yystackp->yytops.yystates[yyk];
      YYASSERT (yyk == 0);
      yystackp->yynextFree -= yynrhs;
      yystackp->yyspaceLeft += yynrhs;
      yystackp->yytops.yystates[0] = & yystackp->yynextFree[-1].yystate;
      YY_REDUCE_PRINT ((1, yyrhs, yyk, yyrule));
      return yyuserAction (yyrule, yynrhs, yyrhs, yystackp,
                           yyvalp, yylocp);
    }
  else
    {
      int yyi;
      yyGLRState* yys;
      yyGLRStackItem yyrhsVals[YYMAXRHS + YYMAXLEFT + 1];
      yys = yyrhsVals[YYMAXRHS + YYMAXLEFT].yystate.yypred
        = yystackp->yytops.yystates[yyk];
      if (yynrhs == 0)
        /* Set default location.  */
        yyrhsVals[YYMAXRHS + YYMAXLEFT - 1].yystate.yyloc = yys->yyloc;
      for (yyi = 0; yyi < yynrhs; yyi += 1)
        {
          yys = yys->yypred;
          YYASSERT (yys);
        }
      yyupdateSplit (yystackp, yys);
      yystackp->yytops.yystates[yyk] = yys;
      YY_REDUCE_PRINT ((0, yyrhsVals + YYMAXRHS + YYMAXLEFT - 1, yyk, yyrule));
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
yyglrReduce (yyGLRStack* yystackp, size_t yyk, yyRuleNum yyrule,
             yybool yyforceEval)
{
  size_t yyposn = yystackp->yytops.yystates[yyk]->yyposn;

  if (yyforceEval || yystackp->yysplitPoint == YY_NULLPTR)
    {
      YYSTYPE yysval;
      YYLTYPE yyloc;

      YYRESULTTAG yyflag = yydoAction (yystackp, yyk, yyrule, &yysval, &yyloc);
      if (yyflag == yyerr && yystackp->yysplitPoint != YY_NULLPTR)
        {
          YYDPRINTF ((stderr, "Parse on stack %lu rejected by rule #%d.\n",
                     (unsigned long int) yyk, yyrule - 1));
        }
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
      size_t yyi;
      int yyn;
      yyGLRState* yys, *yys0 = yystackp->yytops.yystates[yyk];
      yyStateNum yynewLRState;

      for (yys = yystackp->yytops.yystates[yyk], yyn = yyrhsLength (yyrule);
           0 < yyn; yyn -= 1)
        {
          yys = yys->yypred;
          YYASSERT (yys);
        }
      yyupdateSplit (yystackp, yys);
      yynewLRState = yyLRgotoState (yys->yylrState, yylhsNonterm (yyrule));
      YYDPRINTF ((stderr,
                  "Reduced stack %lu by rule #%d; action deferred.  "
                  "Now in state %d.\n",
                  (unsigned long int) yyk, yyrule - 1, yynewLRState));
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
                    YYDPRINTF ((stderr, "Merging stack %lu into stack %lu.\n",
                                (unsigned long int) yyk,
                                (unsigned long int) yyi));
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

static size_t
yysplitStack (yyGLRStack* yystackp, size_t yyk)
{
  if (yystackp->yysplitPoint == YY_NULLPTR)
    {
      YYASSERT (yyk == 0);
      yystackp->yysplitPoint = yystackp->yytops.yystates[yyk];
    }
  if (yystackp->yytops.yysize >= yystackp->yytops.yycapacity)
    {
      yyGLRState** yynewStates;
      yybool* yynewLookaheadNeeds;

      yynewStates = YY_NULLPTR;

      if (yystackp->yytops.yycapacity
          > (YYSIZEMAX / (2 * sizeof yynewStates[0])))
        yyMemoryExhausted (yystackp);
      yystackp->yytops.yycapacity *= 2;

      yynewStates =
        (yyGLRState**) YYREALLOC (yystackp->yytops.yystates,
                                  (yystackp->yytops.yycapacity
                                   * sizeof yynewStates[0]));
      if (yynewStates == YY_NULLPTR)
        yyMemoryExhausted (yystackp);
      yystackp->yytops.yystates = yynewStates;

      yynewLookaheadNeeds =
        (yybool*) YYREALLOC (yystackp->yytops.yylookaheadNeeds,
                             (yystackp->yytops.yycapacity
                              * sizeof yynewLookaheadNeeds[0]));
      if (yynewLookaheadNeeds == YY_NULLPTR)
        yyMemoryExhausted (yystackp);
      yystackp->yytops.yylookaheadNeeds = yynewLookaheadNeeds;
    }
  yystackp->yytops.yystates[yystackp->yytops.yysize]
    = yystackp->yytops.yystates[yyk];
  yystackp->yytops.yylookaheadNeeds[yystackp->yytops.yysize]
    = yystackp->yytops.yylookaheadNeeds[yyk];
  yystackp->yytops.yysize += 1;
  return yystackp->yytops.yysize-1;
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
       yyn > 0;
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
      YYASSERT (yys->yypred);
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
    YYFPRINTF (stderr, "%*s%s -> <Rule %d, empty>\n",
               yyindent, "", yytokenName (yylhsNonterm (yyx->yyrule)),
               yyx->yyrule - 1);
  else
    YYFPRINTF (stderr, "%*s%s -> <Rule %d, tokens %lu .. %lu>\n",
               yyindent, "", yytokenName (yylhsNonterm (yyx->yyrule)),
               yyx->yyrule - 1, (unsigned long int) (yys->yyposn + 1),
               (unsigned long int) yyx->yystate->yyposn);
  for (yyi = 1; yyi <= yynrhs; yyi += 1)
    {
      if (yystates[yyi]->yyresolved)
        {
          if (yystates[yyi-1]->yyposn+1 > yystates[yyi]->yyposn)
            YYFPRINTF (stderr, "%*s%s <empty>\n", yyindent+2, "",
                       yytokenName (yystos[yystates[yyi]->yylrState]));
          else
            YYFPRINTF (stderr, "%*s%s <tokens %lu .. %lu>\n", yyindent+2, "",
                       yytokenName (yystos[yystates[yyi]->yylrState]),
                       (unsigned long int) (yystates[yyi-1]->yyposn + 1),
                       (unsigned long int) yystates[yyi]->yyposn);
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
  YYFPRINTF (stderr, "Ambiguity detected.\n");
  YYFPRINTF (stderr, "Option 1,\n");
  yyreportTree (yyx0, 2);
  YYFPRINTF (stderr, "\nOption 2,\n");
  yyreportTree (yyx1, 2);
  YYFPRINTF (stderr, "\n");
#endif

  yyerror (YY_("syntax is ambiguous"));
  return yyabort;
}

/** Resolve the locations for each of the YYN1 states in *YYSTACKP,
 *  ending at YYS1.  Has no effect on previously resolved states.
 *  The first semantic option of a state is always chosen.  */
static void
yyresolveLocations (yyGLRState* yys1, int yyn1,
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
          YYASSERT (yyoption != YY_NULLPTR);
          yynrhs = yyrhsLength (yyoption->yyrule);
          if (yynrhs > 0)
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
          {
            int yychar_current = yychar;
            YYSTYPE yylval_current = yylval;
            YYLTYPE yylloc_current = yylloc;
            yychar = yyoption->yyrawchar;
            yylval = yyoption->yyval;
            yylloc = yyoption->yyloc;
            YYLLOC_DEFAULT ((yys1->yyloc), yyrhsloc, yynrhs);
            yychar = yychar_current;
            yylval = yylval_current;
            yylloc = yylloc_current;
          }
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
              /* This cannot happen so it is not worth a YYASSERT (yyfalse),
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
  yystackp->yynextFree = ((yyGLRStackItem*) yystackp->yysplitPoint) + 1;
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
yyprocessOneStack (yyGLRStack* yystackp, size_t yyk,
                   size_t yyposn)
{
  while (yystackp->yytops.yystates[yyk] != YY_NULLPTR)
    {
      yyStateNum yystate = yystackp->yytops.yystates[yyk]->yylrState;
      YYDPRINTF ((stderr, "Stack %lu Entering state %d\n",
                  (unsigned long int) yyk, yystate));

      YYASSERT (yystate != YYFINAL);

      if (yyisDefaultedState (yystate))
        {
          YYRESULTTAG yyflag;
          yyRuleNum yyrule = yydefaultAction (yystate);
          if (yyrule == 0)
            {
              YYDPRINTF ((stderr, "Stack %lu dies.\n",
                          (unsigned long int) yyk));
              yymarkStackDeleted (yystackp, yyk);
              return yyok;
            }
          yyflag = yyglrReduce (yystackp, yyk, yyrule, yyimmediate[yyrule]);
          if (yyflag == yyerr)
            {
              YYDPRINTF ((stderr,
                          "Stack %lu dies "
                          "(predicate failure or explicit user error).\n",
                          (unsigned long int) yyk));
              yymarkStackDeleted (yystackp, yyk);
              return yyok;
            }
          if (yyflag != yyok)
            return yyflag;
        }
      else
        {
          yySymbol yytoken;
          int yyaction;
          const short int* yyconflicts;

          yystackp->yytops.yylookaheadNeeds[yyk] = yytrue;
          if (yychar == YYEMPTY)
            {
              YYDPRINTF ((stderr, "Reading a token: "));
              yychar = yylex ();
            }

          if (yychar <= YYEOF)
            {
              yychar = yytoken = YYEOF;
              YYDPRINTF ((stderr, "Now at end of input.\n"));
            }
          else
            {
              yytoken = YYTRANSLATE (yychar);
              YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
            }

          yygetLRActions (yystate, yytoken, &yyaction, &yyconflicts);

          while (*yyconflicts != 0)
            {
              YYRESULTTAG yyflag;
              size_t yynewStack = yysplitStack (yystackp, yyk);
              YYDPRINTF ((stderr, "Splitting off stack %lu from %lu.\n",
                          (unsigned long int) yynewStack,
                          (unsigned long int) yyk));
              yyflag = yyglrReduce (yystackp, yynewStack,
                                    *yyconflicts,
                                    yyimmediate[*yyconflicts]);
              if (yyflag == yyok)
                YYCHK (yyprocessOneStack (yystackp, yynewStack,
                                          yyposn));
              else if (yyflag == yyerr)
                {
                  YYDPRINTF ((stderr, "Stack %lu dies.\n",
                              (unsigned long int) yynewStack));
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
              YYDPRINTF ((stderr, "Stack %lu dies.\n",
                          (unsigned long int) yyk));
              yymarkStackDeleted (yystackp, yyk);
              break;
            }
          else
            {
              YYRESULTTAG yyflag = yyglrReduce (yystackp, yyk, -yyaction,
                                                yyimmediate[-yyaction]);
              if (yyflag == yyerr)
                {
                  YYDPRINTF ((stderr,
                              "Stack %lu dies "
                              "(predicate failure or explicit user error).\n",
                              (unsigned long int) yyk));
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
  size_t yysize0 = yytnamerr (YY_NULLPTR, yytokenName (yytoken));
  size_t yysize = yysize0;
  yybool yysize_overflow = yyfalse;
  char* yymsg = YY_NULLPTR;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected").  */
  int yycount = 0;

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
                  size_t yysz = yysize + yytnamerr (YY_NULLPTR, yytokenName (yyx));
                  yysize_overflow |= yysz < yysize;
                  yysize = yysz;
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
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
#undef YYCASE_
    }

  {
    size_t yysz = yysize + strlen (yyformat);
    yysize_overflow |= yysz < yysize;
    yysize = yysz;
  }

  if (!yysize_overflow)
    yymsg = (char *) YYMALLOC (yysize);

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
              yyp++;
              yyformat++;
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
  size_t yyk;
  int yyj;

  if (yystackp->yyerrState == 3)
    /* We just shifted the error token and (perhaps) took some
       reductions.  Skip tokens until we can proceed.  */
    while (yytrue)
      {
        yySymbol yytoken;
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
          }
        YYDPRINTF ((stderr, "Reading a token: "));
        yychar = yylex ();
        if (yychar <= YYEOF)
          {
            yychar = yytoken = YYEOF;
            YYDPRINTF ((stderr, "Now at end of input.\n"));
          }
        else
          {
            yytoken = YYTRANSLATE (yychar);
            YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
          }
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
  for (yyk = 0; yyk < yystackp->yytops.yysize; yyk += 1)
    if (yystackp->yytops.yystates[yyk] != YY_NULLPTR)
      break;
  if (yyk >= yystackp->yytops.yysize)
    yyFail (yystackp, YY_NULLPTR);
  for (yyk += 1; yyk < yystackp->yytops.yysize; yyk += 1)
    yymarkStackDeleted (yystackp, yyk);
  yyremoveDeletes (yystackp);
  yycompressStack (yystackp);

  /* Now pop stack until we find a state that shifts the error token.  */
  yystackp->yyerrState = 3;
  while (yystackp->yytops.yystates[0] != YY_NULLPTR)
    {
      yyGLRState *yys = yystackp->yytops.yystates[0];
      yyj = yypact[yys->yylrState];
      if (! yypact_value_is_default (yyj))
        {
          yyj += YYTERROR;
          if (0 <= yyj && yyj <= YYLAST && yycheck[yyj] == YYTERROR
              && yyisShiftAction (yytable[yyj]))
            {
              /* Shift the error token.  */
              /* First adjust its location.*/
              YYLTYPE yyerrloc;
              yystackp->yyerror_range[2].yystate.yyloc = yylloc;
              YYLLOC_DEFAULT (yyerrloc, (yystackp->yyerror_range), 2);
              YY_SYMBOL_PRINT ("Shifting", yystos[yytable[yyj]],
                               &yylval, &yyerrloc);
              yyglrShift (yystackp, 0, yytable[yyj],
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
  size_t yyposn;

  YYDPRINTF ((stderr, "Starting parse\n"));

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
          yyRuleNum yyrule;
          int yyaction;
          const short int* yyconflicts;

          yyStateNum yystate = yystack.yytops.yystates[0]->yylrState;
          YYDPRINTF ((stderr, "Entering state %d\n", yystate));
          if (yystate == YYFINAL)
            goto yyacceptlab;
          if (yyisDefaultedState (yystate))
            {
              yyrule = yydefaultAction (yystate);
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
              yySymbol yytoken;
              if (yychar == YYEMPTY)
                {
                  YYDPRINTF ((stderr, "Reading a token: "));
                  yychar = yylex ();
                }

              if (yychar <= YYEOF)
                {
                  yychar = yytoken = YYEOF;
                  YYDPRINTF ((stderr, "Now at end of input.\n"));
                }
              else
                {
                  yytoken = YYTRANSLATE (yychar);
                  YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
                }

              yygetLRActions (yystate, yytoken, &yyaction, &yyconflicts);
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
               yystack.yyerror_range[1].yystate.yyloc = yylloc;
                  yyreportSyntaxError (&yystack);
                  goto yyuser_error;
                }
              else
                YYCHK1 (yyglrReduce (&yystack, 0, -yyaction, yytrue));
            }
        }

      while (yytrue)
        {
          yySymbol yytoken_to_shift;
          size_t yys;

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
              YYDPRINTF ((stderr, "Returning to deterministic operation.\n"));
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
              int yyaction;
              const short int* yyconflicts;
              yyStateNum yystate = yystack.yytops.yystates[yys]->yylrState;
              yygetLRActions (yystate, yytoken_to_shift, &yyaction,
                              &yyconflicts);
              /* Note that yyconflicts were handled by yyprocessOneStack.  */
              YYDPRINTF ((stderr, "On stack %lu, ", (unsigned long int) yys));
              YY_SYMBOL_PRINT ("shifting", yytoken_to_shift, &yylval, &yylloc);
              yyglrShift (&yystack, yys, yyaction, yyposn,
                          &yylval, &yylloc);
              YYDPRINTF ((stderr, "Stack %lu now in state #%d\n",
                          (unsigned long int) yys,
                          yystack.yytops.yystates[yys]->yylrState));
            }

          if (yystack.yytops.yysize == 1)
            {
              YYCHK1 (yyresolveStack (&yystack));
              YYDPRINTF ((stderr, "Returning to deterministic operation.\n"));
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
  YYASSERT (yyfalse);
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
          size_t yysize = yystack.yytops.yysize;
          size_t yyk;
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
      YYFPRINTF (stderr, " -> ");
    }
  YYFPRINTF (stderr, "%d@%lu", yys->yylrState,
             (unsigned long int) yys->yyposn);
}

static void
yypstates (yyGLRState* yyst)
{
  if (yyst == YY_NULLPTR)
    YYFPRINTF (stderr, "<null>");
  else
    yy_yypstack (yyst);
  YYFPRINTF (stderr, "\n");
}

static void
yypstack (yyGLRStack* yystackp, size_t yyk)
{
  yypstates (yystackp->yytops.yystates[yyk]);
}

#define YYINDEX(YYX)                                                         \
    ((YYX) == YY_NULLPTR ? -1 : (yyGLRStackItem*) (YYX) - yystackp->yyitems)


static void
yypdumpstack (yyGLRStack* yystackp)
{
  yyGLRStackItem* yyp;
  size_t yyi;
  for (yyp = yystackp->yyitems; yyp < yystackp->yynextFree; yyp += 1)
    {
      YYFPRINTF (stderr, "%3lu. ",
                 (unsigned long int) (yyp - yystackp->yyitems));
      if (*(yybool *) yyp)
        {
          YYASSERT (yyp->yystate.yyisState);
          YYASSERT (yyp->yyoption.yyisState);
          YYFPRINTF (stderr, "Res: %d, LR State: %d, posn: %lu, pred: %ld",
                     yyp->yystate.yyresolved, yyp->yystate.yylrState,
                     (unsigned long int) yyp->yystate.yyposn,
                     (long int) YYINDEX (yyp->yystate.yypred));
          if (! yyp->yystate.yyresolved)
            YYFPRINTF (stderr, ", firstVal: %ld",
                       (long int) YYINDEX (yyp->yystate
                                             .yysemantics.yyfirstVal));
        }
      else
        {
          YYASSERT (!yyp->yystate.yyisState);
          YYASSERT (!yyp->yyoption.yyisState);
          YYFPRINTF (stderr, "Option. rule: %d, state: %ld, next: %ld",
                     yyp->yyoption.yyrule - 1,
                     (long int) YYINDEX (yyp->yyoption.yystate),
                     (long int) YYINDEX (yyp->yyoption.yynext));
        }
      YYFPRINTF (stderr, "\n");
    }
  YYFPRINTF (stderr, "Tops:");
  for (yyi = 0; yyi < yystackp->yytops.yysize; yyi += 1)
    YYFPRINTF (stderr, "%lu: %ld; ", (unsigned long int) yyi,
               (long int) YYINDEX (yystackp->yytops.yystates[yyi]));
  YYFPRINTF (stderr, "\n");
}
#endif

#undef yylval
#undef yychar
#undef yynerrs
#undef yylloc



#line 787 "bi/parser.ypp" /* glr.c:2584  */

