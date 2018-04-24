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
#define YYFINAL  39
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   466

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  66
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  134
/* YYNRULES -- Number of rules.  */
#define YYNRULES  267
/* YYNRULES -- Number of states.  */
#define YYNSTATES  435
/* YYMAXRHS -- Maximum number of symbols on right-hand side of rule.  */
#define YYMAXRHS 9
/* YYMAXLEFT -- Maximum number of symbols to the left of a handle
   accessed by $0, $-1, etc., in any rule.  */
#define YYMAXLEFT 0

/* YYTRANSLATE(X) -- Bison symbol number corresponding to X.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   297

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    51,     2,     2,     2,     2,    65,     2,
      43,    44,    54,    52,    49,    53,    50,    55,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    58,    60,
      56,    62,    57,    47,    48,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    45,     2,    46,     2,    59,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    63,     2,    64,    61,     2,     2,     2,
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
      35,    36,    37,    38,    39,    40,    41,    42
};

#if YYDEBUG
/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short int yyrline[] =
{
       0,   158,   158,   176,   180,   184,   188,   192,   193,   194,
     195,   199,   203,   207,   208,   212,   216,   220,   224,   228,
     232,   236,   237,   238,   239,   240,   241,   242,   243,   247,
     248,   252,   253,   257,   261,   262,   263,   264,   265,   266,
     267,   271,   272,   276,   277,   278,   282,   283,   287,   288,
     292,   293,   297,   298,   302,   303,   307,   308,   309,   310,
     314,   315,   319,   320,   324,   325,   329,   333,   334,   338,
     342,   343,   347,   351,   352,   356,   357,   361,   362,   363,
     364,   365,   369,   373,   374,   378,   382,   383,   387,   388,
     392,   393,   397,   401,   402,   406,   407,   411,   415,   416,
     420,   421,   425,   426,   430,   431,   435,   436,   440,   441,
     445,   446,   450,   451,   455,   456,   460,   464,   465,   474,
     475,   476,   477,   478,   482,   486,   487,   488,   489,   490,
     494,   494,   498,   498,   502,   502,   506,   506,   510,   510,
     514,   515,   516,   517,   518,   519,   520,   521,   522,   523,
     524,   525,   529,   530,   531,   535,   535,   539,   539,   543,
     544,   545,   546,   550,   550,   554,   554,   558,   558,   559,
     559,   560,   560,   564,   565,   566,   570,   574,   578,   582,
     586,   590,   591,   592,   596,   597,   601,   605,   609,   613,
     617,   621,   625,   626,   627,   628,   629,   630,   631,   632,
     633,   634,   635,   639,   640,   644,   645,   649,   650,   651,
     652,   653,   654,   658,   659,   663,   664,   668,   669,   670,
     671,   672,   673,   674,   675,   676,   677,   678,   682,   683,
     687,   688,   692,   696,   700,   701,   705,   709,   710,   714,
     718,   719,   723,   727,   728,   732,   741,   742,   746,   750,
     754,   758,   762,   766,   767,   768,   769,   770,   771,   775,
     776,   780,   784,   785,   789,   790,   794,   795
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "PROGRAM", "CLASS", "TYPE", "FUNCTION",
  "FIBER", "OPERATOR", "EXPLICIT", "IF", "ELSE", "FOR", "IN", "WHILE",
  "DO", "ASSERT", "RETURN", "YIELD", "CPP", "HPP", "THIS", "SUPER",
  "GLOBAL", "NIL", "DOUBLE_BRACE_OPEN", "DOUBLE_BRACE_CLOSE", "NAME",
  "BOOL_LITERAL", "INT_LITERAL", "REAL_LITERAL", "STRING_LITERAL",
  "LEFT_OP", "RIGHT_OP", "LEFT_TILDE_OP", "RIGHT_TILDE_OP", "AND_OP",
  "OR_OP", "LE_OP", "GE_OP", "EQ_OP", "NE_OP", "RANGE_OP", "'('", "')'",
  "'['", "']'", "'?'", "'@'", "','", "'.'", "'!'", "'+'", "'-'", "'*'",
  "'/'", "'<'", "'>'", "':'", "'_'", "';'", "'~'", "'='", "'{'", "'}'",
  "'&'", "$accept", "name", "bool_literal", "int_literal", "real_literal",
  "string_literal", "literal", "identifier", "parens_expression",
  "sequence_expression", "cast_expression", "function_expression",
  "this_expression", "super_expression", "global_expression",
  "nil_expression", "primary_expression", "index_expression", "index_list",
  "slice", "postfix_expression", "postfix_query_expression",
  "prefix_operator", "prefix_expression", "multiplicative_operator",
  "multiplicative_expression", "additive_operator", "additive_expression",
  "relational_operator", "relational_expression", "equality_operator",
  "equality_expression", "logical_and_operator", "logical_and_expression",
  "logical_or_operator", "logical_or_expression", "expression",
  "optional_expression", "expression_list", "local_variable",
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
  "hpp", "assignment", "expression_statement", "if", "for_index", "for",
  "while", "do_while", "assertion", "return", "yield", "statement",
  "statements", "optional_statements", "class_statement",
  "class_statements", "optional_class_statements", "file_statement",
  "file_statements", "optional_file_statements", "file", "result",
  "optional_result", "value", "optional_value", "braces",
  "optional_braces", "class_braces", "optional_class_braces",
  "double_braces", "weak_modifier", "basic_type", "class_type",
  "unknown_type", "tuple_type", "sequence_type", "postfix_type",
  "function_type", "type", "type_list", "parameter_type_list",
  "parameter_types", YY_NULLPTR
};
#endif

#define YYPACT_NINF -340
#define YYTABLE_NINF -12

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const short int yypact[] =
{
     230,    19,    19,    19,    19,    19,    25,    19,    49,    49,
    -340,    48,  -340,  -340,  -340,  -340,  -340,  -340,  -340,  -340,
    -340,  -340,  -340,   230,  -340,  -340,    83,    69,    61,   104,
      76,    76,    58,    68,   106,  -340,  -340,    70,  -340,  -340,
       6,  -340,     9,  -340,   -13,    19,  -340,    19,    26,    97,
      97,  -340,  -340,  -340,    79,   263,    19,    24,    82,  -340,
      70,    70,   115,    91,  -340,  -340,  -340,   144,  -340,    33,
    -340,   123,   126,   139,   -19,  -340,   152,   135,   148,    19,
    -340,   143,  -340,   150,   151,  -340,   160,   163,    70,  -340,
    -340,  -340,    70,  -340,  -340,  -340,  -340,  -340,  -340,  -340,
    -340,  -340,  -340,  -340,  -340,    19,   182,  -340,   171,   178,
     170,  -340,  -340,   180,   186,   185,   124,    97,  -340,   175,
     155,  -340,  -340,    34,   413,   380,  -340,   181,   183,    70,
    -340,    19,  -340,   369,  -340,  -340,    19,  -340,    19,    68,
    -340,    19,    44,  -340,  -340,  -340,    19,  -340,   -19,   -19,
    -340,   200,    97,  -340,    70,   187,    70,  -340,  -340,  -340,
     196,   204,  -340,  -340,  -340,  -340,  -340,  -340,  -340,  -340,
    -340,  -340,  -340,   413,   316,    76,  -340,  -340,  -340,   203,
      -4,  -340,  -340,  -340,  -340,  -340,  -340,  -340,  -340,  -340,
    -340,  -340,   205,   206,  -340,  -340,   173,  -340,   413,  -340,
      85,    37,    96,   161,   217,   221,  -340,   210,   208,   216,
     218,  -340,   207,  -340,  -340,   214,   220,  -340,  -340,   238,
    -340,   228,   231,   228,   212,   413,   413,   413,   103,    88,
     213,  -340,  -340,  -340,  -340,  -340,  -340,  -340,  -340,  -340,
    -340,  -340,   369,  -340,   215,  -340,  -340,  -340,   222,   233,
    -340,   167,  -340,  -340,  -340,  -340,  -340,    97,  -340,  -340,
    -340,    70,  -340,   234,  -340,   240,    97,   187,    19,    19,
     413,  -340,    19,  -340,  -340,  -340,  -340,  -340,  -340,   413,
    -340,  -340,   413,  -340,  -340,  -340,  -340,   413,  -340,  -340,
     413,  -340,   413,  -340,   413,   413,  -340,  -340,   237,  -340,
     413,  -340,  -340,  -340,   212,    19,   212,   276,   232,  -340,
     235,   236,    70,  -340,  -340,  -340,  -340,  -340,   413,  -340,
    -340,  -340,  -340,  -340,  -340,    19,    19,   112,   239,  -340,
    -340,  -340,  -340,  -340,  -340,   167,  -340,   229,  -340,   -19,
    -340,  -340,  -340,   -19,  -340,  -340,  -340,  -340,   245,   260,
     265,  -340,  -340,    85,    37,    96,   161,   217,  -340,   413,
    -340,   298,   252,  -340,   300,  -340,   228,  -340,  -340,  -340,
     144,    32,   251,    44,    76,    76,    19,  -340,    70,  -340,
    -340,   -19,  -340,  -340,   413,  -340,   413,   270,    17,    70,
     413,   261,   233,  -340,  -340,  -340,  -340,    97,    97,  -340,
     -19,   144,    39,  -340,  -340,  -340,  -340,  -340,  -340,  -340,
     282,  -340,  -340,  -340,  -340,   -19,  -340,    35,  -340,   267,
     268,   413,   -19,   -19,  -340,  -340,   269,  -340,  -340,   286,
    -340,  -340,  -340,   212,  -340
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const unsigned short int yydefact[] =
{
     231,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       2,     0,   217,   218,   219,   220,   221,   222,   223,   224,
     225,   226,   227,   228,   230,   232,     0,     0,   111,     0,
       0,     0,     0,     0,     0,   177,   178,     0,   229,     1,
       0,   134,     0,   110,    89,     0,   175,     0,     0,   235,
     235,   154,   152,   153,     0,     0,     0,     0,     0,   245,
       0,     0,     0,   118,   253,   254,   255,   259,   261,     0,
      93,     0,     0,    95,     0,   104,   109,     0,   106,     0,
      88,   169,   248,     0,     0,    86,     0,    90,     0,   234,
     130,   132,     0,   150,   151,   146,   147,   148,   149,   142,
     143,   140,   141,   144,   145,     0,     0,   112,     0,   114,
     259,   116,   176,   262,     0,     0,     0,   235,   117,   247,
       0,   257,   256,     0,     0,     0,   119,     0,     0,     0,
      94,     0,   241,   206,   240,   135,     0,   105,     0,   118,
     171,     0,     0,   173,   174,    87,     0,   233,     0,     0,
      92,     0,   235,   113,     0,     0,     0,   251,   252,   266,
     264,     0,   260,   246,   250,    17,    18,    19,    20,     3,
       4,     5,     6,     0,     0,     0,    45,    43,    44,   102,
      11,     7,     8,     9,    10,    21,    22,    23,    24,    25,
      26,    27,     0,     0,    28,    34,    41,    46,     0,    50,
      54,    60,    64,    67,    70,    72,    82,    83,     0,     0,
       0,   122,     0,   236,    98,    75,     0,   120,   121,   238,
      96,     0,     0,     0,     0,     0,    74,     0,    11,     0,
       0,   192,   202,   194,   193,   195,   196,   197,   198,   199,
     200,   201,   203,   205,     0,   108,   107,   249,     0,   101,
     244,   216,   243,   170,    91,   131,   133,   235,   157,   115,
     263,     0,   267,     0,    14,     0,   235,     0,     0,     0,
       0,    42,     0,    40,    38,    39,    47,    48,    49,     0,
      52,    53,     0,    58,    59,    56,    57,     0,    62,    63,
       0,    66,     0,    69,     0,     0,    85,   258,     0,   123,
       0,    99,   237,    97,     0,     0,     0,     0,     0,    73,
       0,     0,     0,   159,   160,   161,   180,   162,     0,   124,
     204,   239,   172,   100,   167,     0,     0,     0,     0,   207,
     208,   209,   210,   211,   212,   213,   215,     0,   155,     0,
     265,    12,    13,     0,   103,    11,    35,    36,    31,     0,
      30,    37,    51,    55,    61,    65,    68,    71,    84,     0,
      76,   183,    11,   184,     0,   187,     0,   189,   190,   191,
     259,    77,     0,     0,     0,     0,     0,   165,     0,   214,
     242,     0,   158,    16,     0,    33,     0,     0,     0,     0,
       0,     0,    80,    78,    79,   179,   168,   235,   235,   163,
       0,   259,     0,   156,    32,    29,    15,   182,   181,   185,
       0,   188,    81,   136,   138,     0,   166,     0,   125,     0,
       0,     0,     0,     0,   164,   128,     0,   126,   127,     0,
     137,   139,   129,     0,   186
};

  /* YYPGOTO[NTERM-NUM].  */
static const short int yypgoto[] =
{
    -340,     0,  -340,  -340,  -340,  -340,  -340,  -214,  -203,  -340,
    -340,  -340,  -340,  -340,  -340,  -340,  -340,  -340,   -53,  -340,
    -340,  -340,  -340,  -176,  -340,    50,  -340,    46,  -340,    51,
    -340,    42,  -340,    54,  -340,  -340,   -99,  -340,  -135,  -340,
    -340,    41,  -339,   -18,  -340,   219,   -21,  -340,   211,  -340,
     -94,  -340,    84,  -340,   225,  -340,  -340,   319,   223,  -340,
     295,  -340,  -340,  -340,  -340,  -340,  -340,  -340,  -340,  -340,
    -340,  -340,  -340,  -340,  -340,  -340,  -340,  -340,  -340,  -340,
    -340,  -340,  -340,  -340,  -340,  -340,  -340,  -340,  -340,  -340,
    -340,  -121,  -234,  -340,  -340,   -22,  -340,  -340,  -340,  -340,
    -340,  -340,  -340,  -340,   128,  -340,  -340,    36,  -340,  -340,
     349,  -340,  -340,    47,   -44,  -209,  -340,  -208,  -134,  -340,
       2,   364,  -340,   333,   -55,   246,  -340,  -340,   -28,  -340,
     -29,   -42,   133,  -340
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const short int yydefgoto[] =
{
      -1,   180,   181,   182,   183,   184,   185,   186,   187,   188,
     189,   190,   191,   192,   193,   194,   195,   348,   349,   274,
     196,   197,   198,   199,   279,   200,   282,   201,   287,   202,
     290,   203,   292,   204,   294,   205,   215,   310,   216,   230,
     207,   208,   123,    49,    81,    86,    87,    41,    72,    73,
     127,   324,   209,    43,    77,    78,    44,   118,   108,   109,
     247,    12,   231,   329,    13,   148,    14,   149,    15,    74,
     330,   422,   331,   423,   105,    56,    16,   381,    17,   339,
     318,   332,   415,   333,   400,    18,   373,   142,   248,    19,
      20,    21,    22,   233,   234,   235,   364,   236,   237,   238,
     239,   240,   241,   242,   243,   244,   335,   336,   337,    23,
      24,    25,    26,    89,    90,   128,   303,   134,   135,   252,
     253,    35,   164,    83,   210,    64,    65,    66,   110,    68,
     113,   114,   161,   117
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const short int yytable[] =
{
      11,    27,    28,    29,    30,    31,    91,    33,    69,    67,
     302,    55,   232,    50,   255,   256,   307,   334,   304,   115,
     306,   206,   276,    11,   140,   213,    80,   221,   111,   212,
      48,   392,    54,    10,   229,   106,    10,    63,   263,   265,
      71,   132,    76,   -11,   133,    82,    10,    82,    54,    79,
      70,    10,    57,    10,   346,   347,    54,    63,   351,   147,
      63,    63,   417,   150,   124,   124,    75,    60,    32,    61,
      85,   124,    62,   162,    34,   125,   125,   125,   125,   139,
     133,   107,   125,    39,   151,    10,   249,   160,    63,   280,
     281,   363,    63,   126,   211,   425,   361,    10,   365,   418,
     219,   334,   275,   352,   250,    54,    37,   251,   258,    51,
      52,    53,    40,    60,   260,    61,    63,    42,    62,    48,
     313,   232,   314,   315,    57,   111,   308,   309,   311,    63,
      88,    71,    59,   228,   283,   284,    63,    92,    76,   277,
     278,   139,   112,   229,   376,    88,    54,    57,   316,   317,
     -11,    10,   285,   286,    63,   323,    63,   266,   116,    57,
      45,   312,   394,   391,    46,   360,    47,    60,   159,    61,
     130,   350,    62,   325,   326,   327,   165,   166,   167,   168,
     408,   129,    10,   169,   170,   171,   172,     9,   131,   120,
     136,   121,   137,   420,    10,   122,   206,   138,   173,   141,
     174,   288,   289,   175,   145,   382,   176,   177,   178,   383,
     143,   144,   146,   338,   179,   155,   125,   121,   270,   372,
     271,   122,   343,   272,   273,   434,   152,   154,   153,   156,
     157,   158,   160,     1,     2,     3,     4,     5,     6,     7,
     163,   217,   228,   218,   257,   261,   179,   403,   262,     8,
       9,   328,   267,   291,   296,   268,   269,    10,   293,   295,
     387,    63,   297,   300,   301,   298,   416,   299,   345,   345,
     124,   173,   345,   319,   305,   133,   125,   393,   341,   321,
     359,   424,   322,   371,   370,   350,   342,   405,   430,   431,
     366,   410,   367,   380,   384,   368,   369,   378,   412,    93,
      94,    95,    96,    97,    98,   362,   385,   386,   419,   388,
     389,   395,    63,   390,   406,    99,   100,   101,   102,   103,
     104,   411,   429,   426,   421,   374,   375,   427,   428,   432,
     433,   404,   353,   354,   356,   328,   358,   165,   166,   167,
     168,   355,   220,    10,   169,   170,   171,   172,   357,   402,
     401,   344,    58,   413,   414,   399,   397,   398,   119,   173,
     409,   174,   264,   246,   175,   254,   407,   176,   177,   178,
     320,   379,    38,    36,   377,   396,    54,   259,    63,   221,
      84,   222,   245,   223,   224,   225,   226,   227,     8,    63,
     165,   166,   167,   168,   340,     0,    10,   169,   170,   171,
     172,   165,   166,   167,   168,     0,     0,    10,   169,   170,
     171,   172,   173,     0,   174,     0,     0,   175,     0,     0,
     176,   177,   178,   173,   214,   174,     0,     0,   175,     0,
       0,   176,   177,   178,   165,   166,   167,   168,     0,     0,
      10,   169,   170,   171,   172,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   173,     0,   174,     0,
       0,   175,     0,     0,   176,   177,   178
};

static const short int yycheck[] =
{
       0,     1,     2,     3,     4,     5,    50,     7,    37,    37,
     219,    32,   133,    31,   148,   149,   224,   251,   221,    61,
     223,   120,   198,    23,    79,   124,    44,    10,    57,   123,
      43,   370,    32,    27,   133,    56,    27,    37,   173,   174,
      40,    60,    42,    47,    63,    45,    27,    47,    48,    62,
      44,    27,    56,    27,   268,   269,    56,    57,   272,    88,
      60,    61,   401,    92,    32,    32,    57,    43,    43,    45,
      44,    32,    48,   117,    25,    43,    43,    43,    43,    79,
      63,    57,    43,     0,   105,    27,   141,   116,    88,    52,
      53,   305,    92,    60,    60,    60,   304,    27,   306,    60,
     129,   335,   196,   279,    60,   105,    58,    63,   152,    51,
      52,    53,    43,    43,   156,    45,   116,    56,    48,    43,
      32,   242,    34,    35,    56,   154,   225,   226,   227,   129,
      33,   131,    26,   133,    38,    39,   136,    58,   138,    54,
      55,   141,    60,   242,    32,    33,   146,    56,    60,    61,
      47,    27,    56,    57,   154,   249,   156,   175,    43,    56,
      56,    58,   371,   366,    60,   300,    62,    43,    44,    45,
      44,   270,    48,     6,     7,     8,    21,    22,    23,    24,
     388,    58,    27,    28,    29,    30,    31,    20,    49,    45,
      38,    47,    57,   402,    27,    51,   295,    49,    43,    56,
      45,    40,    41,    48,    44,   339,    51,    52,    53,   343,
      60,    60,    49,   257,    59,    45,    43,    47,    45,   318,
      47,    51,   266,    50,    51,   433,    44,    49,    57,    49,
      44,    46,   261,     3,     4,     5,     6,     7,     8,     9,
      65,    60,   242,    60,    44,    49,    59,   381,    44,    19,
      20,   251,    49,    36,    46,    50,    50,    27,    37,    49,
     359,   261,    46,    49,    44,    47,   400,    60,   268,   269,
      32,    43,   272,    60,    43,    63,    43,   371,    44,    64,
      43,   415,    60,   312,   312,   384,    46,   386,   422,   423,
      14,   390,    60,    64,    49,    60,    60,    58,   392,    36,
      37,    38,    39,    40,    41,   305,    46,    42,   402,    11,
      58,    60,   312,    13,    44,    52,    53,    54,    55,    56,
      57,    60,   421,   417,    42,   325,   326,    60,    60,    60,
      44,   384,   282,   287,   292,   335,   295,    21,    22,    23,
      24,   290,   131,    27,    28,    29,    30,    31,   294,   378,
     378,   267,    33,   397,   398,   376,   374,   375,    63,    43,
     389,    45,    46,   138,    48,   146,   388,    51,    52,    53,
     242,   335,    23,     9,   327,   373,   376,   154,   378,    10,
      47,    12,   136,    14,    15,    16,    17,    18,    19,   389,
      21,    22,    23,    24,   261,    -1,    27,    28,    29,    30,
      31,    21,    22,    23,    24,    -1,    -1,    27,    28,    29,
      30,    31,    43,    -1,    45,    -1,    -1,    48,    -1,    -1,
      51,    52,    53,    43,    44,    45,    -1,    -1,    48,    -1,
      -1,    51,    52,    53,    21,    22,    23,    24,    -1,    -1,
      27,    28,    29,    30,    31,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    43,    -1,    45,    -1,
      -1,    48,    -1,    -1,    51,    52,    53
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,     3,     4,     5,     6,     7,     8,     9,    19,    20,
      27,    67,   127,   130,   132,   134,   142,   144,   151,   155,
     156,   157,   158,   175,   176,   177,   178,    67,    67,    67,
      67,    67,    43,    67,    25,   187,   187,    58,   176,     0,
      43,   113,    56,   119,   122,    56,    60,    62,    43,   109,
     109,    51,    52,    53,    67,   112,   141,    56,   123,    26,
      43,    45,    48,    67,   191,   192,   193,   194,   195,   196,
      44,    67,   114,   115,   135,    57,    67,   120,   121,    62,
     109,   110,    67,   189,   189,    44,   111,   112,    33,   179,
     180,   180,    58,    36,    37,    38,    39,    40,    41,    52,
      53,    54,    55,    56,    57,   140,   112,    57,   124,   125,
     194,   196,    60,   196,   197,   197,    43,   199,   123,   126,
      45,    47,    51,   108,    32,    43,    60,   116,   181,    58,
      44,    49,    60,    63,   183,   184,    38,    57,    49,    67,
     190,    56,   153,    60,    60,    44,    49,   196,   131,   133,
     196,   112,    44,    57,    49,    45,    49,    44,    46,    44,
     196,   198,   180,    65,   188,    21,    22,    23,    24,    28,
      29,    30,    31,    43,    45,    48,    51,    52,    53,    59,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    86,    87,    88,    89,
      91,    93,    95,    97,    99,   101,   102,   106,   107,   118,
     190,    60,   116,   102,    44,   102,   104,    60,    60,   196,
     114,    10,    12,    14,    15,    16,    17,    18,    67,   102,
     105,   128,   157,   159,   160,   161,   163,   164,   165,   166,
     167,   168,   169,   170,   171,   191,   120,   126,   154,   190,
      60,    63,   185,   186,   111,   184,   184,    44,   180,   124,
     197,    49,    44,   104,    46,   104,   109,    49,    50,    50,
      45,    47,    50,    51,    85,   116,    89,    54,    55,    90,
      52,    53,    92,    38,    39,    56,    57,    94,    40,    41,
      96,    36,    98,    37,   100,    49,    46,    46,    47,    60,
      49,    44,   181,   182,    74,    43,    74,   183,   102,   102,
     103,   102,    58,    32,    34,    35,    60,    61,   146,    60,
     170,    64,    60,   116,   117,     6,     7,     8,    67,   129,
     136,   138,   147,   149,   158,   172,   173,   174,   180,   145,
     198,    44,    46,   180,   118,    67,    73,    73,    83,    84,
     102,    73,    89,    91,    93,    95,    97,    99,   107,    43,
     104,   183,    67,    73,   162,   183,    14,    60,    60,    60,
     194,   196,   102,   152,    67,    67,    32,   179,    58,   173,
      64,   143,   184,   184,    49,    46,    42,   102,    11,    58,
      13,    74,   108,   116,   181,    60,   186,   109,   109,   112,
     150,   194,   196,   184,    84,   102,    44,   161,   183,   196,
     102,    60,   116,   180,   180,   148,   184,   108,    60,   116,
     181,    42,   137,   139,   184,    60,   116,    60,    60,   102,
     184,   184,    60,    44,   183
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,    66,    67,    68,    69,    70,    71,    72,    72,    72,
      72,    73,    74,    75,    75,    76,    77,    78,    79,    80,
      81,    82,    82,    82,    82,    82,    82,    82,    82,    83,
      83,    84,    84,    85,    86,    86,    86,    86,    86,    86,
      86,    87,    87,    88,    88,    88,    89,    89,    90,    90,
      91,    91,    92,    92,    93,    93,    94,    94,    94,    94,
      95,    95,    96,    96,    97,    97,    98,    99,    99,   100,
     101,   101,   102,   103,   103,   104,   104,   105,   105,   105,
     105,   105,   106,   107,   107,   108,   109,   109,   110,   110,
     111,   111,   112,   113,   113,   114,   114,   115,   116,   116,
     117,   117,   118,   118,   119,   119,   120,   120,   121,   121,
     122,   122,   123,   123,   124,   124,   125,   126,   126,   127,
     127,   127,   127,   127,   128,   129,   129,   129,   129,   129,
     131,   130,   133,   132,   135,   134,   137,   136,   139,   138,
     140,   140,   140,   140,   140,   140,   140,   140,   140,   140,
     140,   140,   141,   141,   141,   143,   142,   145,   144,   146,
     146,   146,   146,   148,   147,   150,   149,   152,   151,   153,
     151,   154,   151,   155,   155,   155,   156,   157,   158,   159,
     160,   161,   161,   161,   162,   162,   163,   164,   165,   166,
     167,   168,   169,   169,   169,   169,   169,   169,   169,   169,
     169,   169,   169,   170,   170,   171,   171,   172,   172,   172,
     172,   172,   172,   173,   173,   174,   174,   175,   175,   175,
     175,   175,   175,   175,   175,   175,   175,   175,   176,   176,
     177,   177,   178,   179,   180,   180,   181,   182,   182,   183,
     184,   184,   185,   186,   186,   187,   188,   188,   189,   190,
     191,   192,   193,   194,   194,   194,   194,   194,   194,   195,
     195,   196,   197,   197,   198,   198,   199,   199
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const unsigned char yyr2[] =
{
       0,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     3,     3,     2,     5,     4,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     3,
       1,     1,     3,     3,     1,     3,     3,     3,     2,     2,
       2,     1,     2,     1,     1,     1,     1,     2,     1,     1,
       1,     3,     1,     1,     1,     3,     1,     1,     1,     1,
       1,     3,     1,     1,     1,     3,     1,     1,     3,     1,
       1,     3,     1,     1,     0,     1,     3,     3,     4,     4,
       4,     5,     1,     1,     3,     3,     2,     3,     1,     0,
       1,     3,     3,     2,     3,     1,     3,     4,     2,     3,
       1,     0,     1,     3,     2,     3,     1,     3,     3,     1,
       1,     0,     2,     3,     1,     3,     1,     1,     0,     4,
       5,     5,     5,     6,     2,     4,     5,     5,     5,     6,
       0,     6,     0,     6,     0,     5,     0,     6,     0,     6,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     0,     9,     0,     8,     1,
       1,     1,     1,     0,     5,     0,     4,     0,     9,     0,
       6,     0,     7,     5,     5,     3,     4,     2,     2,     4,
       2,     5,     5,     3,     1,     3,     9,     3,     5,     3,
       3,     3,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     2,     1,     0,     1,     1,     1,
       1,     1,     1,     1,     2,     1,     0,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       1,     0,     1,     2,     1,     0,     2,     1,     0,     3,
       1,     1,     3,     1,     1,     2,     1,     0,     1,     2,
       3,     3,     3,     1,     1,     1,     2,     2,     4,     1,
       3,     1,     1,     3,     1,     3,     2,     3
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
       0,     0,     0,     0,     0,     0,     0,     0
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
       0,     0,     0,     0,     0,     0,     0,     0
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
       0,     0,     0,     0,     0,     0,     0,     0
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
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     5,     0,     0,     0,     0,     0,     0,
       0,     0,     7,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     1,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     3,     0,     0,
       9,     0,     0,     0,     0,     0,     0,     0,     0,    11,
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
       0,     0,     0,     0,     0,     0,     0
};

/* YYCONFL[I] -- lists of conflicting rule numbers, each terminated by
   0, pointed into by YYCONFLP.  */
static const short int yyconfl[] =
{
       0,   111,     0,   118,     0,   118,     0,    11,     0,   118,
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
#line 158 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valString)); }
#line 1418 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 3:
#line 176 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Literal<bool>((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valBool), yytext, new bi::BasicType(new bi::Name("Boolean")), make_loc((*yylocp))); }
#line 1424 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 4:
#line 180 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Literal<int64_t>((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valInt), yytext, new bi::BasicType(new bi::Name("Integer")), make_loc((*yylocp))); }
#line 1430 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 5:
#line 184 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Literal<double>((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valReal), yytext, new bi::BasicType(new bi::Name("Real")), make_loc((*yylocp))); }
#line 1436 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 6:
#line 188 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Literal<const char*>((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valString), yytext, new bi::BasicType(new bi::Name("String")), make_loc((*yylocp))); }
#line 1442 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 11:
#line 199 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Identifier<bi::Unknown>((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valName), make_loc((*yylocp))); }
#line 1448 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 12:
#line 203 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Parentheses((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1454 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 13:
#line 207 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Sequence((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1460 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 14:
#line 208 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Sequence(empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1466 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 15:
#line 212 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Cast(new bi::PointerType(false, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1472 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 16:
#line 216 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LambdaFunction((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 1478 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 17:
#line 220 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::This(make_loc((*yylocp))); }
#line 1484 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 18:
#line 224 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Super(make_loc((*yylocp))); }
#line 1490 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 19:
#line 228 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Global(make_loc((*yylocp))); }
#line 1496 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 20:
#line 232 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Nil(make_loc((*yylocp))); }
#line 1502 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 29:
#line 247 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Range((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1508 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 30:
#line 248 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Index((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1514 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 32:
#line 253 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1520 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 33:
#line 257 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1526 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 35:
#line 262 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Member((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1532 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 36:
#line 263 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Member((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1538 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 37:
#line 264 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Member((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1544 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 38:
#line 265 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Slice((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1550 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 39:
#line 266 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Call((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1556 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 40:
#line 267 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Get((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1562 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 42:
#line 272 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Query((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1568 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 43:
#line 276 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::UnaryOperator>(new bi::Name("+"), make_loc((*yylocp))); }
#line 1574 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 44:
#line 277 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::UnaryOperator>(new bi::Name("-"), make_loc((*yylocp))); }
#line 1580 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 45:
#line 278 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::UnaryOperator>(new bi::Name("!"), make_loc((*yylocp))); }
#line 1586 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 47:
#line 283 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::UnaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1592 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 48:
#line 287 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("*"), make_loc((*yylocp))); }
#line 1598 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 49:
#line 288 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("/"), make_loc((*yylocp))); }
#line 1604 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 51:
#line 293 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), make_loc((*yylocp))); }
#line 1610 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 52:
#line 297 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("+"), make_loc((*yylocp))); }
#line 1616 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 53:
#line 298 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("-"), make_loc((*yylocp))); }
#line 1622 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 55:
#line 303 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), make_loc((*yylocp))); }
#line 1628 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 56:
#line 307 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("<"), make_loc((*yylocp))); }
#line 1634 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 57:
#line 308 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name(">"), make_loc((*yylocp))); }
#line 1640 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 58:
#line 309 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("<="), make_loc((*yylocp))); }
#line 1646 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 59:
#line 310 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name(">="), make_loc((*yylocp))); }
#line 1652 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 61:
#line 315 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), make_loc((*yylocp))); }
#line 1658 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 62:
#line 319 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("=="), make_loc((*yylocp))); }
#line 1664 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 63:
#line 320 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("!="), make_loc((*yylocp))); }
#line 1670 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 65:
#line 325 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), make_loc((*yylocp))); }
#line 1676 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 66:
#line 329 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("&&"), make_loc((*yylocp))); }
#line 1682 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 68:
#line 334 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), make_loc((*yylocp))); }
#line 1688 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 69:
#line 338 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("||"), make_loc((*yylocp))); }
#line 1694 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 71:
#line 343 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), make_loc((*yylocp))); }
#line 1700 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 74:
#line 352 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1706 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 76:
#line 357 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1712 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 77:
#line 361 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LocalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1718 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 78:
#line 362 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LocalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1724 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 79:
#line 363 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LocalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1730 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 80:
#line 364 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LocalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1736 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 81:
#line 365 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LocalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1742 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 82:
#line 369 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Span((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1748 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 84:
#line 374 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1754 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 85:
#line 378 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1760 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 86:
#line 382 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1766 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 87:
#line 383 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1772 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 89:
#line 388 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1778 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 91:
#line 393 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1784 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 92:
#line 397 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Parameter((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1790 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 93:
#line 401 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1796 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 94:
#line 402 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1802 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 96:
#line 407 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1808 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 97:
#line 411 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Parameter((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1814 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 98:
#line 415 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1820 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 99:
#line 416 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1826 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 101:
#line 421 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1832 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 102:
#line 425 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valInt) = 1; }
#line 1838 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 103:
#line 426 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valInt) = (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valInt) + 1; }
#line 1844 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 104:
#line 430 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1850 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 105:
#line 431 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1856 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 107:
#line 436 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1862 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 108:
#line 440 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Generic((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 1868 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 109:
#line 441 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Generic((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valName), new bi::AnyType(make_loc((*yylocp))), make_loc((*yylocp))); }
#line 1874 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 111:
#line 446 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1880 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 112:
#line 450 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 1886 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 113:
#line 451 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType); }
#line 1892 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 115:
#line 456 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::TypeList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 1898 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 118:
#line 465 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 1904 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 119:
#line 474 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1910 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 120:
#line 475 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1916 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 121:
#line 476 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 1922 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 122:
#line 477 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1928 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 123:
#line 478 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1934 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 124:
#line 482 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::ExpressionStatement((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 1940 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 125:
#line 486 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1946 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 126:
#line 487 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1952 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 127:
#line 488 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 1958 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 128:
#line 489 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1964 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 129:
#line 490 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1970 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 130:
#line 494 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 1976 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 131:
#line 494 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Function(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 1982 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 132:
#line 498 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 1988 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 133:
#line 498 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Fiber(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 1994 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 134:
#line 502 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2000 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 135:
#line 502 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Program((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2006 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 136:
#line 506 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2012 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 137:
#line 506 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::MemberFunction(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2018 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 138:
#line 510 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2024 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 139:
#line 510 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::MemberFiber(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2030 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 140:
#line 514 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('*'); }
#line 2036 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 141:
#line 515 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('/'); }
#line 2042 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 142:
#line 516 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('+'); }
#line 2048 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 143:
#line 517 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('-'); }
#line 2054 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 144:
#line 518 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('<'); }
#line 2060 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 145:
#line 519 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('>'); }
#line 2066 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 146:
#line 520 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("<="); }
#line 2072 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 147:
#line 521 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name(">="); }
#line 2078 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 148:
#line 522 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("=="); }
#line 2084 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 149:
#line 523 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("!="); }
#line 2090 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 150:
#line 524 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("&&"); }
#line 2096 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 151:
#line 525 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("||"); }
#line 2102 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 152:
#line 529 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('+'); }
#line 2108 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 153:
#line 530 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('-'); }
#line 2114 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 154:
#line 531 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('!'); }
#line 2120 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 155:
#line 535 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2126 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 156:
#line 535 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::BinaryOperator((((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2132 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 157:
#line 539 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2138 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 158:
#line 539 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::UnaryOperator((((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2144 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 159:
#line 543 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("<-"); }
#line 2150 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 160:
#line 544 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("<~"); }
#line 2156 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 161:
#line 545 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("~>"); }
#line 2162 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 162:
#line 546 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("~"); }
#line 2168 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 163:
#line 550 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2174 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 164:
#line 550 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::AssignmentOperator(new bi::Name("<-"), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2180 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 165:
#line 554 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2186 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 166:
#line 554 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::ConversionOperator((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2192 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 167:
#line 558 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2198 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 168:
#line 558 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Class((((yyGLRStackItem const *)yyvsp)[YYFILL (-7)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), false, (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2204 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 169:
#line 559 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2210 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 170:
#line 559 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Class((((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), empty_type((*yylocp)), false, empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2216 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 171:
#line 560 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2222 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 172:
#line 560 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Class((((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), true, empty_expr((*yylocp)), empty_stmt((*yylocp)), make_doc_loc((*yylocp))); }
#line 2228 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 173:
#line 564 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Basic((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), false, make_doc_loc((*yylocp))); }
#line 2234 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 174:
#line 565 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Basic((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), true, make_doc_loc((*yylocp))); }
#line 2240 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 175:
#line 566 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Basic((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), empty_type((*yylocp)), false, make_doc_loc((*yylocp))); }
#line 2246 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 176:
#line 570 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Explicit(new bi::ClassType((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))), make_doc_loc((*yylocp))); }
#line 2252 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 177:
#line 574 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Raw(new bi::Name("cpp"), pop_raw(), make_loc((*yylocp))); }
#line 2258 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 178:
#line 578 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Raw(new bi::Name("hpp"), pop_raw(), make_loc((*yylocp))); }
#line 2264 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 179:
#line 582 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Assignment((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2270 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 180:
#line 586 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::ExpressionStatement((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2276 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 181:
#line 590 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::If((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2282 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 182:
#line 591 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::If((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2288 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 183:
#line 592 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::If((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), empty_stmt((*yylocp)), make_loc((*yylocp))); }
#line 2294 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 185:
#line 597 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LocalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 2300 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 186:
#line 601 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::For((((yyGLRStackItem const *)yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2306 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 187:
#line 605 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::While((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2312 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 188:
#line 609 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::DoWhile((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2318 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 189:
#line 613 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Assert((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2324 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 190:
#line 617 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Return((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2330 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 191:
#line 621 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Yield((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2336 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 204:
#line 640 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::StatementList((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2342 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 206:
#line 645 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2348 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 214:
#line 659 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::StatementList((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2354 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 216:
#line 664 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2360 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 229:
#line 683 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::StatementList((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2366 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 231:
#line 688 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2372 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 232:
#line 692 "bi/parser.ypp" /* glr.c:816  */
    { compiler->setRoot((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement)); }
#line 2378 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 233:
#line 696 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType); }
#line 2384 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 235:
#line 701 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 2390 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 236:
#line 705 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression); }
#line 2396 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 238:
#line 710 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 2402 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 239:
#line 714 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Braces((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2408 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 241:
#line 719 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2414 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 242:
#line 723 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Braces((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2420 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 244:
#line 728 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2426 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 246:
#line 741 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valBool) = true; }
#line 2432 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 247:
#line 742 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valBool) = false; }
#line 2438 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 248:
#line 746 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::BasicType((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valName), make_loc((*yylocp))); }
#line 2444 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 249:
#line 750 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::ClassType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2450 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 250:
#line 754 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::UnknownType((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valBool), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2456 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 251:
#line 758 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::TupleType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2462 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 252:
#line 762 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::SequenceType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2468 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 256:
#line 769 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::FiberType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2474 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 257:
#line 770 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::OptionalType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2480 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 258:
#line 771 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::ArrayType((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valInt), make_loc((*yylocp))); }
#line 2486 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 260:
#line 776 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::FunctionType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2492 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 263:
#line 785 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::TypeList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2498 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 265:
#line 790 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::TypeList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2504 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 266:
#line 794 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 2510 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 267:
#line 795 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType); }
#line 2516 "bi/parser.cpp" /* glr.c:816  */
    break;


#line 2520 "bi/parser.cpp" /* glr.c:816  */
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
  (!!((Yystate) == (-340)))

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



#line 798 "bi/parser.ypp" /* glr.c:2584  */

