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
#define YYFINAL  41
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   638

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  68
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  138
/* YYNRULES -- Number of rules.  */
#define YYNRULES  276
/* YYNRULES -- Number of states.  */
#define YYNSTATES  458
/* YYMAXRHS -- Maximum number of symbols on right-hand side of rule.  */
#define YYMAXRHS 10
/* YYMAXLEFT -- Maximum number of symbols to the left of a handle
   accessed by $0, $-1, etc., in any rule.  */
#define YYMAXLEFT 0

/* YYTRANSLATE(X) -- Bison symbol number corresponding to X.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   299

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    53,     2,     2,     2,     2,    67,     2,
      45,    46,    56,    54,    51,    55,    52,    57,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    60,    62,
      58,    64,    59,    49,    50,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    47,     2,    48,     2,    61,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    65,     2,    66,    63,     2,     2,     2,
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
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44
};

#if YYDEBUG
/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short int yyrline[] =
{
       0,   160,   160,   169,   173,   177,   181,   185,   186,   187,
     188,   192,   196,   200,   201,   205,   209,   213,   217,   221,
     225,   229,   230,   231,   232,   233,   234,   235,   236,   240,
     241,   245,   246,   250,   254,   255,   256,   257,   258,   259,
     260,   264,   265,   269,   270,   271,   275,   276,   280,   281,
     285,   286,   290,   291,   295,   296,   300,   301,   302,   303,
     307,   308,   312,   313,   317,   318,   322,   326,   327,   331,
     335,   336,   340,   341,   345,   346,   350,   354,   355,   359,
     360,   364,   365,   366,   367,   368,   369,   373,   377,   378,
     382,   386,   387,   391,   392,   396,   397,   401,   405,   406,
     410,   411,   415,   419,   420,   424,   425,   429,   430,   434,
     435,   439,   440,   444,   448,   449,   453,   454,   458,   459,
     463,   467,   468,   477,   478,   479,   480,   481,   482,   486,
     490,   491,   492,   493,   494,   495,   499,   499,   503,   503,
     507,   507,   511,   511,   515,   515,   519,   520,   521,   522,
     523,   524,   525,   526,   527,   528,   529,   530,   534,   535,
     536,   540,   540,   544,   544,   548,   549,   553,   553,   557,
     557,   561,   561,   562,   562,   563,   563,   567,   568,   569,
     573,   577,   581,   585,   589,   593,   594,   595,   599,   600,
     604,   605,   609,   610,   614,   618,   622,   626,   630,   634,
     635,   636,   637,   638,   639,   640,   641,   642,   643,   644,
     648,   649,   653,   654,   658,   659,   660,   661,   662,   663,
     667,   668,   672,   673,   677,   678,   679,   680,   681,   682,
     683,   684,   685,   686,   687,   691,   692,   696,   697,   701,
     705,   709,   710,   714,   718,   719,   723,   727,   728,   732,
     736,   737,   741,   750,   751,   755,   759,   763,   767,   768,
     772,   776,   780,   781,   782,   783,   784,   785,   789,   790,
     794,   798,   799,   803,   804,   808,   809
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "PROGRAM", "CLASS", "TYPE", "FUNCTION",
  "FIBER", "OPERATOR", "AUTO", "EXPLICIT", "IF", "ELSE", "FOR", "IN",
  "WHILE", "DO", "ASSERT", "RETURN", "YIELD", "CPP", "HPP", "THIS",
  "SUPER", "GLOBAL", "PARALLEL", "NIL", "DOUBLE_BRACE_OPEN",
  "DOUBLE_BRACE_CLOSE", "NAME", "BOOL_LITERAL", "INT_LITERAL",
  "REAL_LITERAL", "STRING_LITERAL", "LEFT_OP", "RIGHT_OP", "LEFT_TILDE_OP",
  "RIGHT_TILDE_OP", "AND_OP", "OR_OP", "LE_OP", "GE_OP", "EQ_OP", "NE_OP",
  "RANGE_OP", "'('", "')'", "'['", "']'", "'?'", "'@'", "','", "'.'",
  "'!'", "'+'", "'-'", "'*'", "'/'", "'<'", "'>'", "':'", "'_'", "';'",
  "'~'", "'='", "'{'", "'}'", "'&'", "$accept", "name", "bool_literal",
  "int_literal", "real_literal", "string_literal", "literal", "identifier",
  "parens_expression", "sequence_expression", "cast_expression",
  "function_expression", "this_expression", "super_expression",
  "global_expression", "nil_expression", "primary_expression",
  "index_expression", "index_list", "slice", "postfix_expression",
  "postfix_query_expression", "prefix_operator", "prefix_expression",
  "multiplicative_operator", "multiplicative_expression",
  "additive_operator", "additive_expression", "relational_operator",
  "relational_expression", "equality_operator", "equality_expression",
  "logical_and_operator", "logical_and_expression", "logical_or_operator",
  "logical_or_expression", "assign_operator", "assign_expression",
  "expression", "optional_expression", "expression_list", "local_variable",
  "span_expression", "span_list", "brackets", "parameters",
  "optional_parameters", "parameter_list", "parameter", "options",
  "option_list", "option", "arguments", "optional_arguments", "size",
  "generics", "generic_list", "generic", "optional_generics",
  "generic_arguments", "generic_argument_list", "generic_argument",
  "optional_generic_arguments", "global_variable_declaration",
  "local_variable_declaration", "member_variable_declaration",
  "function_declaration", "$@1", "fiber_declaration", "$@2",
  "program_declaration", "$@3", "member_function_declaration", "$@4",
  "member_fiber_declaration", "$@5", "binary_operator", "unary_operator",
  "binary_operator_declaration", "$@6", "unary_operator_declaration",
  "$@7", "assignment_operator", "assignment_operator_declaration", "$@8",
  "conversion_operator_declaration", "$@9", "class_declaration", "$@10",
  "$@11", "$@12", "basic_declaration", "explicit_declaration", "cpp",
  "hpp", "assignment", "expression_statement", "if", "for_annotation",
  "for_index", "for", "while", "do_while", "assertion", "return", "yield",
  "statement", "statements", "optional_statements", "class_statement",
  "class_statements", "optional_class_statements", "file_statement",
  "file_statements", "optional_file_statements", "file", "result",
  "optional_result", "value", "optional_value", "braces",
  "optional_braces", "class_braces", "optional_class_braces",
  "double_braces", "weak_modifier", "basic_type", "class_type",
  "unknown_type", "member_type", "tuple_type", "sequence_type",
  "postfix_type", "function_type", "type", "type_list",
  "parameter_type_list", "parameter_types", YY_NULLPTR
};
#endif

#define YYPACT_NINF -358
#define YYTABLE_NINF -214

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const short int yypact[] =
{
     166,    30,    30,    30,    30,    30,    28,    30,    30,    80,
      80,  -358,    49,  -358,  -358,  -358,  -358,  -358,  -358,  -358,
    -358,  -358,  -358,  -358,   166,  -358,  -358,   120,    96,    89,
     151,   111,   111,    46,   128,   109,   150,  -358,  -358,    98,
    -358,  -358,     8,  -358,     3,  -358,    -7,    30,  -358,    30,
      48,   145,   145,  -358,  -358,  -358,   122,    75,    30,   583,
     127,     1,   132,  -358,    98,    98,   167,   156,  -358,   168,
    -358,  -358,    25,  -358,    21,  -358,   169,   181,   177,    84,
    -358,  -358,   171,   180,    30,  -358,   176,  -358,   170,   173,
    -358,   190,   187,    98,  -358,  -358,  -358,    98,  -358,  -358,
    -358,  -358,  -358,  -358,  -358,  -358,  -358,  -358,  -358,  -358,
      30,   195,  -358,  -358,  -358,  -358,  -358,  -358,  -358,  -358,
     583,   515,   111,  -358,  -358,  -358,   -13,  -358,  -358,  -358,
    -358,  -358,  -358,  -358,  -358,  -358,  -358,  -358,   192,   193,
    -358,  -358,   152,  -358,   583,  -358,    83,    27,   125,   174,
     205,    -5,  -358,  -358,  -358,  -358,   188,   197,    59,  -358,
    -358,   198,   200,   202,    76,   145,  -358,   184,    30,   481,
    -358,  -358,    90,   549,  -358,   194,   196,    98,  -358,    30,
    -358,   391,  -358,  -358,  -358,    30,   109,  -358,    30,   138,
    -358,  -358,  -358,    30,  -358,    84,    84,  -358,   206,   145,
     204,   213,  -358,   212,   145,   214,    30,    30,   583,  -358,
      30,  -358,  -358,  -358,  -358,  -358,  -358,   583,  -358,  -358,
     583,  -358,  -358,  -358,  -358,   583,  -358,  -358,   583,  -358,
     583,  -358,  -358,  -358,   583,   583,  -358,    98,   203,    98,
    -358,  -358,  -358,   210,   219,  -358,  -358,  -358,  -358,   218,
    -358,   220,   222,   224,  -358,   211,  -358,   228,  -358,  -358,
     128,  -358,    30,   583,   583,   216,   583,   583,   583,  -358,
     101,     6,   215,  -358,  -358,  -358,  -358,  -358,   262,  -358,
    -358,  -358,  -358,  -358,  -358,   443,  -358,   217,  -358,  -358,
     225,   233,  -358,    82,  -358,  -358,  -358,  -358,  -358,   145,
    -358,   583,  -358,  -358,    84,   234,  -358,  -358,  -358,   231,
     236,   242,  -358,  -358,    83,    27,   125,   174,   205,  -358,
    -358,  -358,    98,  -358,   203,   583,  -358,  -358,  -358,  -358,
    -358,  -358,   128,   216,   216,   273,   227,  -358,   229,   230,
      98,  -358,  -358,  -358,   583,  -358,    51,  -358,  -358,  -358,
    -358,  -358,    30,    30,   191,    30,   235,  -358,  -358,  -358,
    -358,  -358,  -358,    82,  -358,   232,  -358,    84,  -358,  -358,
     583,   583,  -358,   583,  -358,  -358,  -358,  -358,   282,  -358,
     583,  -358,  -358,  -358,    25,   110,   238,    30,   237,  -358,
     287,   138,   111,   111,    30,  -358,   128,    98,  -358,  -358,
      84,  -358,   256,  -358,  -358,    14,   241,   233,  -358,  -358,
    -358,   290,    98,   583,  -358,   145,   145,  -358,    84,   244,
      25,    42,  -358,  -358,  -358,  -358,  -358,  -358,   583,  -358,
     263,  -358,  -358,    84,  -358,  -358,    92,  -358,   246,   247,
     267,   583,    84,    84,  -358,  -358,   250,  -358,  -358,   583,
     216,  -358,  -358,  -358,   269,  -358,   216,  -358
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const unsigned short int yydefact[] =
{
     238,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     2,     0,   224,   225,   226,   227,   228,   229,   230,
     231,   232,   233,   234,   235,   237,   239,     0,     0,   115,
       0,     0,     0,     0,     0,     0,     0,   181,   182,     0,
     236,     1,     0,   140,     0,   114,    94,     0,   179,     0,
       0,   242,   242,   160,   158,   159,     0,     0,     0,     0,
       0,     0,     0,   252,     0,     0,     0,   122,   258,   262,
     263,   264,   268,   270,     0,    98,     0,     0,   100,     0,
     109,   113,     0,   111,     0,    93,   173,   255,     0,     0,
      91,     0,    95,     0,   241,   136,   138,     0,   156,   157,
     152,   153,   154,   155,   148,   149,   146,   147,   150,   151,
       0,     0,    17,    18,    19,    20,     3,     4,     5,     6,
       0,     0,     0,    45,    43,    44,    11,     7,     8,     9,
      10,    21,    22,    23,    24,    25,    26,    27,     0,     0,
      28,    34,    41,    46,     0,    50,    54,    60,    64,    67,
      70,    74,    76,   243,   126,   116,     0,   118,   268,   120,
     180,   271,     0,     0,     0,   242,   121,   254,     0,     0,
     266,   265,     0,     0,   123,     0,     0,     0,    99,     0,
     248,   188,   247,   141,   110,     0,   122,   175,     0,     0,
     177,   178,    92,     0,   240,     0,     0,    97,     0,   242,
      79,     0,    14,     0,   242,     0,     0,     0,     0,    42,
       0,    40,    38,    39,    47,    48,    49,     0,    52,    53,
       0,    58,    59,    56,    57,     0,    62,    63,     0,    66,
       0,    72,    73,    69,     0,     0,   117,     0,     0,     0,
     260,   261,   275,   273,     0,   269,   253,   257,   259,   107,
      87,    88,     0,     0,   127,     0,   103,     0,   124,   125,
     245,   101,     0,     0,     0,     0,     0,    78,     0,   189,
      11,     0,     0,   199,   209,   201,   200,   202,     0,   203,
     204,   205,   206,   207,   208,   188,   212,     0,   112,   256,
       0,   106,   251,   223,   250,   174,    96,   137,   139,   242,
     163,     0,    12,    13,     0,     0,    11,    35,    36,    31,
       0,    30,    37,    51,    55,    61,    65,    68,    71,    75,
     119,   272,     0,   276,     0,     0,    90,   267,   128,   104,
     244,   102,     0,     0,     0,     0,     0,    77,     0,     0,
       0,   165,   184,   166,     0,   129,     0,   211,   246,   176,
     105,   171,     0,     0,     0,     0,     0,   214,   215,   216,
     217,   218,   219,   220,   222,     0,   161,     0,    80,    16,
       0,     0,    33,     0,   274,   108,    89,    84,   187,   194,
       0,   196,   197,   198,   268,    81,     0,     0,    11,   190,
       0,     0,     0,     0,     0,   169,     0,     0,   221,   249,
       0,   164,     0,    32,    29,     0,     0,    85,    82,    83,
     183,     0,     0,     0,   172,   242,   242,   167,     0,     0,
     268,     0,   162,    15,   186,   185,   195,    86,     0,   191,
       0,   142,   144,     0,   170,   133,     0,   130,     0,     0,
       0,     0,     0,     0,   168,   134,     0,   131,   132,     0,
       0,   143,   145,   135,     0,   193,     0,   192
};

  /* YYPGOTO[NTERM-NUM].  */
static const short int yypgoto[] =
{
    -358,     0,  -358,  -358,  -358,  -358,  -358,  -154,  -358,  -358,
    -358,  -358,  -358,  -358,  -358,  -358,  -358,  -358,   -58,  -358,
    -358,  -358,  -358,  -122,  -358,    97,  -358,    91,  -358,    95,
    -358,    88,  -358,    85,  -358,  -358,  -358,    94,   -45,  -358,
    -103,  -358,  -358,    -1,  -357,   -20,  -358,   134,   -18,  -358,
     141,  -358,  -131,  -358,     7,  -358,   148,  -358,  -358,   295,
      99,  -358,   -44,  -358,  -358,  -358,  -358,  -358,  -358,  -358,
    -358,  -358,  -358,  -358,  -358,  -358,  -358,  -358,  -358,  -358,
    -358,  -358,  -358,  -358,  -358,  -358,  -358,  -358,  -358,  -358,
    -358,  -358,  -358,  -160,  -265,  -358,  -358,   -71,  -358,   -50,
    -358,  -358,  -358,  -358,  -358,  -358,  -358,    53,  -358,  -358,
     -24,  -358,  -358,   317,  -358,  -358,   -12,   -46,   -64,  -358,
    -248,  -176,  -358,   -48,   334,  -358,   296,   159,   182,  -358,
    -358,  -358,   -30,  -358,   -26,   -49,    26,  -358
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const short int yydefgoto[] =
{
      -1,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   309,   310,   212,
     142,   143,   144,   145,   217,   146,   220,   147,   225,   148,
     228,   149,   230,   150,   234,   151,   235,   152,   200,   338,
     201,   272,   251,   252,   172,    51,    86,    91,    92,    43,
      77,    78,   175,   351,   253,    45,    82,    83,    46,   166,
     156,   157,   205,    13,   273,   357,    14,   195,    15,   196,
      16,    79,   358,   442,   359,   443,   110,    58,    17,   400,
      18,   367,   344,   360,   433,   361,   418,    19,   391,   189,
     290,    20,    21,    22,    23,   275,   276,   277,   278,   390,
     279,   280,   281,   282,   283,   284,   285,   286,   287,   363,
     364,   365,    24,    25,    26,    27,    94,    95,    60,   331,
     182,   183,   294,   295,    37,   247,    88,   187,    68,    69,
      70,    71,   158,    73,   161,   162,   244,   165
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const short int yytable[] =
{
      12,    28,    29,    30,    31,    32,    96,    34,    35,    72,
     176,   213,    52,    74,   153,    57,   163,   335,   203,   297,
     298,   274,   214,   167,    12,   263,    85,   407,   362,   231,
      11,   232,    11,    56,   233,   159,   -11,    11,    50,    67,
     111,   255,    76,   341,    81,    61,    64,    87,    65,    87,
      56,    66,   307,   308,    75,    59,   312,    84,    56,    11,
     155,    67,    80,   436,    67,    67,   173,   194,   342,   343,
     257,   197,   169,    33,   170,    11,    59,    11,   171,   181,
      11,   218,   219,   174,   186,   378,   379,   173,   352,   353,
     354,   355,   198,    67,    90,   313,   387,    67,   362,    53,
      54,    55,   204,    10,   437,    11,   238,    36,   170,    39,
      56,    11,   171,    98,    99,   100,   101,   102,   103,   245,
      41,    64,   242,    65,   250,   274,    66,    11,   369,   104,
     105,   106,   107,   108,   109,   173,   271,   173,   243,   215,
     216,    42,   289,    64,    59,    65,   180,    44,    66,   181,
     -11,   260,   254,   300,   445,   173,    50,   425,   304,    61,
     350,   340,    59,   311,    67,   221,   222,    61,    67,     1,
       2,     3,     4,     5,     6,     7,     8,    67,    63,    76,
      93,   270,    97,   223,   224,    81,     9,    10,   186,   154,
     321,   401,   389,    56,   160,    11,   330,   173,   368,   208,
     292,   209,   455,   293,   210,   211,   306,   306,   457,    47,
     306,   159,   164,    48,    61,    49,   226,   227,   333,   334,
     168,   336,   337,   339,   422,   394,    93,   178,   179,   177,
     184,   185,   190,   389,   188,   191,   192,    67,   193,    67,
     271,   199,   434,   229,   206,   207,   240,   236,   237,   239,
     241,   246,   299,   366,   408,   301,   258,   444,   259,   302,
     303,   322,   332,   305,   249,   323,   451,   452,   377,   324,
     326,   325,   327,   328,   329,   346,   427,   345,   173,   370,
     250,   181,   371,   348,   372,   270,   373,   349,   380,   381,
     438,   382,   383,   356,   405,   397,   243,   412,   399,   386,
     410,   413,   423,   426,   428,   446,   435,   441,   447,   448,
     384,   449,   453,   403,   385,   456,   315,   314,   317,   318,
     261,   409,    67,   316,   376,   402,   311,   296,   404,   319,
      62,   375,   419,   288,   424,   406,   320,   411,   347,   398,
      67,    40,   395,   414,    38,    89,   388,   291,   374,     0,
     248,     0,   392,   393,     0,   396,     0,   439,     0,     0,
       0,     0,     0,   356,     0,     0,     0,   420,   430,   431,
     432,   421,   415,   416,     0,     0,   417,     0,     0,     0,
       0,     0,     0,   440,     0,     0,   429,   388,     0,     0,
       0,     0,     0,     0,    56,     0,   450,    67,     0,     0,
     262,     0,   263,     0,   454,     0,   264,   265,   266,   267,
     268,     9,    67,   112,   113,   114,   269,   115,     0,     0,
      11,   116,   117,   118,   119,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   120,     0,   121,     0,
       0,   122,     0,     0,   123,   124,   125,     0,     0,     0,
       0,     0,   262,     0,   263,     0,     0,  -213,   264,   265,
     266,   267,   268,     9,     0,   112,   113,   114,   269,   115,
       0,     0,    11,   116,   117,   118,   119,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   120,     0,
     121,     0,     0,   122,     0,     0,   123,   124,   125,     0,
       0,     0,     0,   112,   113,   114,     0,   115,     0,  -210,
      11,   116,   117,   118,   119,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   120,     0,   121,     0,
       0,   122,     0,     0,   123,   124,   125,   112,   113,   114,
       0,   115,   249,     0,    11,   116,   117,   118,   119,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     120,     0,   121,   202,     0,   122,     0,     0,   123,   124,
     125,   112,   113,   114,     0,   115,     0,     0,    11,   116,
     117,   118,   119,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   120,   256,   121,     0,     0,   122,
       0,     0,   123,   124,   125,   112,   113,   114,     0,   115,
       0,     0,    11,   116,   117,   118,   119,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   120,     0,
     121,     0,     0,   122,     0,     0,   123,   124,   125
};

static const short int yycheck[] =
{
       0,     1,     2,     3,     4,     5,    52,     7,     8,    39,
      74,   142,    32,    39,    59,    33,    65,   265,   121,   195,
     196,   181,   144,    67,    24,    11,    46,   384,   293,    34,
      29,    36,    29,    33,    39,    61,    49,    29,    45,    39,
      58,   172,    42,    37,    44,    58,    45,    47,    47,    49,
      50,    50,   206,   207,    46,    34,   210,    64,    58,    29,
      59,    61,    59,   420,    64,    65,    45,    93,    62,    63,
     173,    97,    47,    45,    49,    29,    34,    29,    53,    65,
      29,    54,    55,    62,    84,   333,   334,    45,     6,     7,
       8,     9,   110,    93,    46,   217,    45,    97,   363,    53,
      54,    55,   122,    21,    62,    29,    47,    27,    49,    60,
     110,    29,    53,    38,    39,    40,    41,    42,    43,   165,
       0,    45,    46,    47,   169,   285,    50,    29,   304,    54,
      55,    56,    57,    58,    59,    45,   181,    45,   164,    56,
      57,    45,   186,    45,    34,    47,    62,    58,    50,    65,
      49,   177,    62,   199,    62,    45,    45,   405,   204,    58,
     291,    60,    34,   208,   164,    40,    41,    58,   168,     3,
       4,     5,     6,     7,     8,     9,    10,   177,    28,   179,
      35,   181,    60,    58,    59,   185,    20,    21,   188,    62,
     239,   367,   346,   193,    62,    29,   260,    45,   301,    47,
      62,    49,   450,    65,    52,    53,   206,   207,   456,    58,
     210,   237,    45,    62,    58,    64,    42,    43,   263,   264,
      52,   266,   267,   268,   400,    34,    35,    46,    51,    60,
      59,    51,    62,   387,    58,    62,    46,   237,    51,   239,
     285,    46,   418,    38,    52,    52,    46,    59,    51,    51,
      48,    67,    46,   299,   385,    51,    62,   433,    62,    46,
      48,    51,   262,    49,    61,    46,   442,   443,   332,    51,
      48,    51,    48,    62,    46,    13,   407,    62,    45,    45,
     325,    65,    51,    66,    48,   285,    44,    62,    15,    62,
     421,    62,    62,   293,    12,    60,   322,    60,    66,   344,
      62,    14,    46,    62,    14,   436,    62,    44,    62,    62,
     340,    44,    62,   371,   340,    46,   225,   220,   230,   234,
     179,   385,   322,   228,   325,   370,   371,   193,   373,   235,
      35,   324,   396,   185,   405,   380,   237,   387,   285,   363,
     340,    24,   354,   391,    10,    49,   346,   188,   322,    -1,
     168,    -1,   352,   353,    -1,   355,    -1,   421,    -1,    -1,
      -1,    -1,    -1,   363,    -1,    -1,    -1,   397,   413,   415,
     416,   397,   392,   393,    -1,    -1,   394,    -1,    -1,    -1,
      -1,    -1,    -1,   428,    -1,    -1,   412,   387,    -1,    -1,
      -1,    -1,    -1,    -1,   394,    -1,   441,   397,    -1,    -1,
       9,    -1,    11,    -1,   449,    -1,    15,    16,    17,    18,
      19,    20,   412,    22,    23,    24,    25,    26,    -1,    -1,
      29,    30,    31,    32,    33,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    -1,
      -1,    50,    -1,    -1,    53,    54,    55,    -1,    -1,    -1,
      -1,    -1,     9,    -1,    11,    -1,    -1,    66,    15,    16,
      17,    18,    19,    20,    -1,    22,    23,    24,    25,    26,
      -1,    -1,    29,    30,    31,    32,    33,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    45,    -1,
      47,    -1,    -1,    50,    -1,    -1,    53,    54,    55,    -1,
      -1,    -1,    -1,    22,    23,    24,    -1,    26,    -1,    66,
      29,    30,    31,    32,    33,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    -1,
      -1,    50,    -1,    -1,    53,    54,    55,    22,    23,    24,
      -1,    26,    61,    -1,    29,    30,    31,    32,    33,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      45,    -1,    47,    48,    -1,    50,    -1,    -1,    53,    54,
      55,    22,    23,    24,    -1,    26,    -1,    -1,    29,    30,
      31,    32,    33,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    45,    46,    47,    -1,    -1,    50,
      -1,    -1,    53,    54,    55,    22,    23,    24,    -1,    26,
      -1,    -1,    29,    30,    31,    32,    33,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    45,    -1,
      47,    -1,    -1,    50,    -1,    -1,    53,    54,    55
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,     3,     4,     5,     6,     7,     8,     9,    10,    20,
      21,    29,    69,   131,   134,   136,   138,   146,   148,   155,
     159,   160,   161,   162,   180,   181,   182,   183,    69,    69,
      69,    69,    69,    45,    69,    69,    27,   192,   192,    60,
     181,     0,    45,   117,    58,   123,   126,    58,    62,    64,
      45,   113,   113,    53,    54,    55,    69,   116,   145,    34,
     186,    58,   127,    28,    45,    47,    50,    69,   196,   197,
     198,   199,   200,   201,   202,    46,    69,   118,   119,   139,
      59,    69,   124,   125,    64,   113,   114,    69,   194,   194,
      46,   115,   116,    35,   184,   185,   185,    60,    38,    39,
      40,    41,    42,    43,    54,    55,    56,    57,    58,    59,
     144,   116,    22,    23,    24,    26,    30,    31,    32,    33,
      45,    47,    50,    53,    54,    55,    69,    70,    71,    72,
      73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    88,    89,    90,    91,    93,    95,    97,    99,
     101,   103,   105,   106,    62,    59,   128,   129,   200,   202,
      62,   202,   203,   203,    45,   205,   127,   130,    52,    47,
      49,    53,   112,    45,    62,   120,   186,    60,    46,    51,
      62,    65,   188,   189,    59,    51,    69,   195,    58,   157,
      62,    62,    46,    51,   202,   135,   137,   202,   116,    46,
     106,   108,    48,   108,   113,   130,    52,    52,    47,    49,
      52,    53,    87,   120,    91,    56,    57,    92,    54,    55,
      94,    40,    41,    58,    59,    96,    42,    43,    98,    38,
     100,    34,    36,    39,   102,   104,    59,    51,    47,    51,
      46,    48,    46,   202,   204,   185,    67,   193,   196,    61,
     106,   110,   111,   122,    62,   120,    46,   108,    62,    62,
     202,   118,     9,    11,    15,    16,    17,    18,    19,    25,
      69,   106,   109,   132,   161,   163,   164,   165,   166,   168,
     169,   170,   171,   172,   173,   174,   175,   176,   124,   130,
     158,   195,    62,    65,   190,   191,   115,   189,   189,    46,
     185,    51,    46,    48,   185,    49,    69,    75,    75,    85,
      86,   106,    75,    91,    93,    95,    97,    99,   101,   105,
     128,   203,    51,    46,    51,    51,    48,    48,    62,    46,
     186,   187,    69,   106,   106,   188,   106,   106,   107,   106,
      60,    37,    62,    63,   150,    62,    13,   175,    66,    62,
     120,   121,     6,     7,     8,     9,    69,   133,   140,   142,
     151,   153,   162,   177,   178,   179,   185,   149,   108,   189,
      45,    51,    48,    44,   204,   122,   111,   186,   188,   188,
      15,    62,    62,    62,   200,   202,   106,    45,    69,    75,
     167,   156,    69,    69,    34,   184,    69,    60,   178,    66,
     147,   189,   106,    86,   106,    12,   106,   112,   120,   186,
      62,   167,    60,    14,   191,   113,   113,   116,   154,   186,
     200,   202,   189,    46,   165,   188,    62,   120,    14,   202,
     106,   185,   185,   152,   189,    62,   112,    62,   120,   186,
     106,    44,   141,   143,   189,    62,   120,    62,    62,    44,
     106,   189,   189,    62,   106,   188,    46,   188
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,    68,    69,    70,    71,    72,    73,    74,    74,    74,
      74,    75,    76,    77,    77,    78,    79,    80,    81,    82,
      83,    84,    84,    84,    84,    84,    84,    84,    84,    85,
      85,    86,    86,    87,    88,    88,    88,    88,    88,    88,
      88,    89,    89,    90,    90,    90,    91,    91,    92,    92,
      93,    93,    94,    94,    95,    95,    96,    96,    96,    96,
      97,    97,    98,    98,    99,    99,   100,   101,   101,   102,
     103,   103,   104,   104,   105,   105,   106,   107,   107,   108,
     108,   109,   109,   109,   109,   109,   109,   110,   111,   111,
     112,   113,   113,   114,   114,   115,   115,   116,   117,   117,
     118,   118,   119,   120,   120,   121,   121,   122,   122,   123,
     123,   124,   124,   125,   126,   126,   127,   127,   128,   128,
     129,   130,   130,   131,   131,   131,   131,   131,   131,   132,
     133,   133,   133,   133,   133,   133,   135,   134,   137,   136,
     139,   138,   141,   140,   143,   142,   144,   144,   144,   144,
     144,   144,   144,   144,   144,   144,   144,   144,   145,   145,
     145,   147,   146,   149,   148,   150,   150,   152,   151,   154,
     153,   156,   155,   157,   155,   158,   155,   159,   159,   159,
     160,   161,   162,   163,   164,   165,   165,   165,   166,   166,
     167,   167,   168,   168,   169,   170,   171,   172,   173,   174,
     174,   174,   174,   174,   174,   174,   174,   174,   174,   174,
     175,   175,   176,   176,   177,   177,   177,   177,   177,   177,
     178,   178,   179,   179,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   181,   181,   182,   182,   183,
     184,   185,   185,   186,   187,   187,   188,   189,   189,   190,
     191,   191,   192,   193,   193,   194,   195,   196,   197,   197,
     198,   199,   200,   200,   200,   200,   200,   200,   201,   201,
     202,   203,   203,   204,   204,   205,   205
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const unsigned char yyr2[] =
{
       0,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     3,     3,     2,     6,     4,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     3,
       1,     1,     3,     3,     1,     3,     3,     3,     2,     2,
       2,     1,     2,     1,     1,     1,     1,     2,     1,     1,
       1,     3,     1,     1,     1,     3,     1,     1,     1,     1,
       1,     3,     1,     1,     1,     3,     1,     1,     3,     1,
       1,     3,     1,     1,     1,     3,     1,     1,     0,     1,
       3,     3,     4,     4,     3,     4,     5,     1,     1,     3,
       3,     2,     3,     1,     0,     1,     3,     3,     2,     3,
       1,     3,     4,     2,     3,     1,     0,     1,     3,     2,
       3,     1,     3,     1,     1,     0,     2,     3,     1,     3,
       1,     1,     0,     4,     5,     5,     4,     5,     6,     2,
       4,     5,     5,     4,     5,     6,     0,     6,     0,     6,
       0,     5,     0,     6,     0,     6,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     0,     9,     0,     8,     1,     1,     0,     5,     0,
       4,     0,     9,     0,     6,     0,     7,     5,     5,     3,
       4,     2,     2,     4,     2,     5,     5,     3,     0,     1,
       1,     3,    10,     8,     3,     5,     3,     3,     3,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     2,     1,     0,     1,     1,     1,     1,     1,     1,
       1,     2,     1,     0,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     2,     1,     0,     1,
       2,     1,     0,     2,     1,     0,     3,     1,     1,     3,
       1,     1,     2,     1,     0,     1,     2,     3,     1,     3,
       3,     3,     1,     1,     1,     2,     2,     4,     1,     3,
       1,     1,     3,     1,     3,     2,     3
};


/* YYDPREC[RULE-NUM] -- Dynamic precedence of rule #RULE-NUM (0 if none).  */
static const unsigned char yydprec[] =
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
       0,     0,     0,     0,     0,     0,     0
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
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0
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
       0,     0,     0,     0,     0,     0,     0
};

/* YYCONFLP[YYPACT[STATE-NUM]] -- Pointer into YYCONFL of start of
   list of conflicting reductions corresponding to action entry for
   state STATE-NUM in yytable.  0 means no conflicts.  The list in
   yyconfl is terminated by a rule number of 0.  */
static const unsigned char yyconflp[] =
{
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     5,     0,     0,     0,
       0,     0,     0,     0,     0,     7,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     1,     0,     0,
       9,     0,     0,     0,     0,     0,     0,     0,     0,    11,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     3,     0,     0,     0,     0,     0,
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
static const short int yyconfl[] =
{
       0,   115,     0,   122,     0,   122,     0,    11,     0,   122,
       0,    11,     0
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
#line 1482 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 3:
#line 169 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Literal<bool>((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valBool), yytext, new bi::BasicType(new bi::Name("Boolean")), make_loc((*yylocp))); }
#line 1488 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 4:
#line 173 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Literal<int64_t>((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valInt), yytext, new bi::BasicType(new bi::Name("Integer")), make_loc((*yylocp))); }
#line 1494 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 5:
#line 177 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Literal<double>((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valReal), yytext, new bi::BasicType(new bi::Name("Real")), make_loc((*yylocp))); }
#line 1500 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 6:
#line 181 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Literal<const char*>((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valString), yytext, new bi::BasicType(new bi::Name("String")), make_loc((*yylocp))); }
#line 1506 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 11:
#line 192 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Identifier<bi::Unknown>((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valName), make_loc((*yylocp))); }
#line 1512 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 12:
#line 196 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Parentheses((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1518 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 13:
#line 200 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Sequence((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1524 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 14:
#line 201 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Sequence(empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1530 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 15:
#line 205 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Cast(new bi::UnknownType(false, (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1536 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 16:
#line 209 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LambdaFunction((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 1542 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 17:
#line 213 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::This(make_loc((*yylocp))); }
#line 1548 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 18:
#line 217 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Super(make_loc((*yylocp))); }
#line 1554 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 19:
#line 221 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Global(make_loc((*yylocp))); }
#line 1560 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 20:
#line 225 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Nil(make_loc((*yylocp))); }
#line 1566 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 29:
#line 240 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Range((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1572 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 30:
#line 241 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Index((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1578 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 32:
#line 246 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1584 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 33:
#line 250 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1590 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 35:
#line 255 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Member((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1596 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 36:
#line 256 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Member((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1602 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 37:
#line 257 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Member((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1608 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 38:
#line 258 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Slice((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1614 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 39:
#line 259 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Call((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1620 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 40:
#line 260 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Get((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1626 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 42:
#line 265 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Query((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1632 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 43:
#line 269 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::UnaryOperator>(new bi::Name("+"), make_loc((*yylocp))); }
#line 1638 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 44:
#line 270 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::UnaryOperator>(new bi::Name("-"), make_loc((*yylocp))); }
#line 1644 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 45:
#line 271 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::UnaryOperator>(new bi::Name("!"), make_loc((*yylocp))); }
#line 1650 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 47:
#line 276 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::UnaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1656 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 48:
#line 280 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("*"), make_loc((*yylocp))); }
#line 1662 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 49:
#line 281 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("/"), make_loc((*yylocp))); }
#line 1668 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 51:
#line 286 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), make_loc((*yylocp))); }
#line 1674 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 52:
#line 290 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("+"), make_loc((*yylocp))); }
#line 1680 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 53:
#line 291 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("-"), make_loc((*yylocp))); }
#line 1686 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 55:
#line 296 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), make_loc((*yylocp))); }
#line 1692 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 56:
#line 300 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("<"), make_loc((*yylocp))); }
#line 1698 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 57:
#line 301 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name(">"), make_loc((*yylocp))); }
#line 1704 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 58:
#line 302 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("<="), make_loc((*yylocp))); }
#line 1710 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 59:
#line 303 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name(">="), make_loc((*yylocp))); }
#line 1716 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 61:
#line 308 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), make_loc((*yylocp))); }
#line 1722 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 62:
#line 312 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("=="), make_loc((*yylocp))); }
#line 1728 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 63:
#line 313 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("!="), make_loc((*yylocp))); }
#line 1734 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 65:
#line 318 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), make_loc((*yylocp))); }
#line 1740 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 66:
#line 322 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("&&"), make_loc((*yylocp))); }
#line 1746 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 68:
#line 327 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), make_loc((*yylocp))); }
#line 1752 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 69:
#line 331 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("||"), make_loc((*yylocp))); }
#line 1758 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 71:
#line 336 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), make_loc((*yylocp))); }
#line 1764 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 72:
#line 340 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("<-"); }
#line 1770 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 73:
#line 341 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("<~"); }
#line 1776 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 75:
#line 346 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Assign((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1782 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 78:
#line 355 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1788 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 80:
#line 360 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1794 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 81:
#line 364 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LocalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1800 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 82:
#line 365 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LocalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1806 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 83:
#line 366 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LocalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1812 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 84:
#line 367 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LocalVariable(bi::AUTO, (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), empty_type((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1818 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 85:
#line 368 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LocalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1824 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 86:
#line 369 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LocalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1830 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 87:
#line 373 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Span((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1836 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 89:
#line 378 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1842 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 90:
#line 382 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1848 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 91:
#line 386 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1854 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 92:
#line 387 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1860 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 94:
#line 392 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1866 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 96:
#line 397 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1872 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 97:
#line 401 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Parameter((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1878 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 98:
#line 405 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1884 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 99:
#line 406 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1890 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 101:
#line 411 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1896 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 102:
#line 415 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Parameter((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1902 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 103:
#line 419 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1908 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 104:
#line 420 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1914 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 106:
#line 425 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1920 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 107:
#line 429 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valInt) = 1; }
#line 1926 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 108:
#line 430 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valInt) = (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valInt) + 1; }
#line 1932 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 109:
#line 434 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1938 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 110:
#line 435 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1944 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 112:
#line 440 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1950 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 113:
#line 444 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Generic((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valName), empty_type((*yylocp)), make_loc((*yylocp))); }
#line 1956 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 115:
#line 449 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1962 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 116:
#line 453 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 1968 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 117:
#line 454 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType); }
#line 1974 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 119:
#line 459 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::TypeList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 1980 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 122:
#line 468 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 1986 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 123:
#line 477 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1992 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 124:
#line 478 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1998 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 125:
#line 479 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 2004 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 126:
#line 480 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::AUTO, (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), empty_type((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2010 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 127:
#line 481 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2016 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 128:
#line 482 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2022 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 129:
#line 486 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::ExpressionStatement((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 2028 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 130:
#line 490 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2034 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 131:
#line 491 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2040 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 132:
#line 492 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 2046 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 133:
#line 493 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::AUTO, (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), empty_type((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2052 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 134:
#line 494 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2058 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 135:
#line 495 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2064 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 136:
#line 499 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2070 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 137:
#line 499 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Function(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2076 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 138:
#line 503 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2082 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 139:
#line 503 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Fiber(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), new bi::FiberType((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2088 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 140:
#line 507 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2094 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 141:
#line 507 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Program((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2100 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 142:
#line 511 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2106 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 143:
#line 511 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::MemberFunction(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2112 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 144:
#line 515 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2118 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 145:
#line 515 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::MemberFiber(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), new bi::FiberType((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2124 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 146:
#line 519 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('*'); }
#line 2130 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 147:
#line 520 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('/'); }
#line 2136 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 148:
#line 521 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('+'); }
#line 2142 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 149:
#line 522 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('-'); }
#line 2148 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 150:
#line 523 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('<'); }
#line 2154 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 151:
#line 524 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('>'); }
#line 2160 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 152:
#line 525 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("<="); }
#line 2166 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 153:
#line 526 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name(">="); }
#line 2172 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 154:
#line 527 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("=="); }
#line 2178 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 155:
#line 528 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("!="); }
#line 2184 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 156:
#line 529 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("&&"); }
#line 2190 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 157:
#line 530 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("||"); }
#line 2196 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 158:
#line 534 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('+'); }
#line 2202 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 159:
#line 535 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('-'); }
#line 2208 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 160:
#line 536 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('!'); }
#line 2214 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 161:
#line 540 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2220 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 162:
#line 540 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::BinaryOperator((((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2226 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 163:
#line 544 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2232 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 164:
#line 544 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::UnaryOperator((((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2238 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 165:
#line 548 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("~>"); }
#line 2244 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 166:
#line 549 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("~"); }
#line 2250 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 167:
#line 553 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2256 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 168:
#line 553 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::AssignmentOperator((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2262 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 169:
#line 557 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2268 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 170:
#line 557 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::ConversionOperator((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2274 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 171:
#line 561 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2280 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 172:
#line 561 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Class((((yyGLRStackItem const *)yyvsp)[YYFILL (-7)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), false, (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2286 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 173:
#line 562 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2292 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 174:
#line 562 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Class((((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), empty_type((*yylocp)), false, empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2298 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 175:
#line 563 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2304 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 176:
#line 563 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Class((((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), true, empty_expr((*yylocp)), empty_stmt((*yylocp)), make_doc_loc((*yylocp))); }
#line 2310 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 177:
#line 567 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Basic((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), false, make_doc_loc((*yylocp))); }
#line 2316 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 178:
#line 568 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Basic((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), true, make_doc_loc((*yylocp))); }
#line 2322 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 179:
#line 569 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Basic((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), empty_type((*yylocp)), false, make_doc_loc((*yylocp))); }
#line 2328 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 180:
#line 573 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Explicit(new bi::ClassType((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))), make_doc_loc((*yylocp))); }
#line 2334 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 181:
#line 577 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Raw(new bi::Name("cpp"), pop_raw(), make_loc((*yylocp))); }
#line 2340 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 182:
#line 581 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Raw(new bi::Name("hpp"), pop_raw(), make_loc((*yylocp))); }
#line 2346 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 183:
#line 585 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Assignment((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2352 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 184:
#line 589 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::ExpressionStatement((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2358 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 185:
#line 593 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::If((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2364 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 186:
#line 594 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::If((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2370 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 187:
#line 595 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::If((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), empty_stmt((*yylocp)), make_loc((*yylocp))); }
#line 2376 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 188:
#line 599 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valAnnotation) = bi::NONE; }
#line 2382 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 189:
#line 600 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valAnnotation) = bi::PARALLEL; }
#line 2388 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 191:
#line 605 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LocalVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 2394 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 192:
#line 609 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::For((((yyGLRStackItem const *)yyvsp)[YYFILL (-9)].yystate.yysemantics.yysval.valAnnotation), (((yyGLRStackItem const *)yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2400 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 193:
#line 610 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::For((((yyGLRStackItem const *)yyvsp)[YYFILL (-7)].yystate.yysemantics.yysval.valAnnotation), (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2406 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 194:
#line 614 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::While((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2412 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 195:
#line 618 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::DoWhile((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2418 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 196:
#line 622 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Assert((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2424 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 197:
#line 626 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Return((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2430 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 198:
#line 630 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Yield((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2436 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 211:
#line 649 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::StatementList((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2442 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 213:
#line 654 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2448 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 221:
#line 668 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::StatementList((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2454 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 223:
#line 673 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2460 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 236:
#line 692 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::StatementList((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2466 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 238:
#line 697 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2472 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 239:
#line 701 "bi/parser.ypp" /* glr.c:816  */
    { compiler->setRoot((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement)); }
#line 2478 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 240:
#line 705 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType); }
#line 2484 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 242:
#line 710 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 2490 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 243:
#line 714 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression); }
#line 2496 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 245:
#line 719 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 2502 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 246:
#line 723 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Braces((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2508 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 248:
#line 728 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2514 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 249:
#line 732 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Braces((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2520 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 251:
#line 737 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2526 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 253:
#line 750 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valBool) = true; }
#line 2532 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 254:
#line 751 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valBool) = false; }
#line 2538 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 255:
#line 755 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::BasicType((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valName), make_loc((*yylocp))); }
#line 2544 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 256:
#line 759 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::ClassType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2550 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 257:
#line 763 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::UnknownType((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valBool), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2556 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 259:
#line 768 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::MemberType((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2562 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 260:
#line 772 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::TupleType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2568 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 261:
#line 776 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::SequenceType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2574 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 265:
#line 783 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::FiberType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2580 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 266:
#line 784 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::OptionalType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2586 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 267:
#line 785 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::ArrayType((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valInt), make_loc((*yylocp))); }
#line 2592 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 269:
#line 790 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::FunctionType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2598 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 272:
#line 799 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::TypeList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2604 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 274:
#line 804 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::TypeList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2610 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 275:
#line 808 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 2616 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 276:
#line 809 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType); }
#line 2622 "bi/parser.cpp" /* glr.c:816  */
    break;


#line 2626 "bi/parser.cpp" /* glr.c:816  */
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
  (!!((Yystate) == (-358)))

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



#line 812 "bi/parser.ypp" /* glr.c:2584  */

