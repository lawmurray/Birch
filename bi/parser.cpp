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
#define YYLAST   565

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  67
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  137
/* YYNRULES -- Number of rules.  */
#define YYNRULES  271
/* YYNRULES -- Number of states.  */
#define YYNSTATES  444
/* YYMAXRHS -- Maximum number of symbols on right-hand side of rule.  */
#define YYMAXRHS 10
/* YYMAXLEFT -- Maximum number of symbols to the left of a handle
   accessed by $0, $-1, etc., in any rule.  */
#define YYMAXLEFT 0

/* YYTRANSLATE(X) -- Bison symbol number corresponding to X.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   298

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    52,     2,     2,     2,     2,    66,     2,
      44,    45,    55,    53,    50,    54,    51,    56,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    59,    61,
      57,    63,    58,    48,    49,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    46,     2,    47,     2,    60,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    64,     2,    65,    62,     2,     2,     2,
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
      35,    36,    37,    38,    39,    40,    41,    42,    43
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
     360,   364,   365,   366,   367,   368,   372,   376,   377,   381,
     385,   386,   390,   391,   395,   396,   400,   404,   405,   409,
     410,   414,   418,   419,   423,   424,   428,   429,   433,   434,
     438,   439,   443,   447,   448,   452,   453,   457,   458,   462,
     466,   467,   476,   477,   478,   479,   480,   484,   488,   489,
     490,   491,   492,   496,   496,   500,   500,   504,   504,   508,
     508,   512,   512,   516,   517,   518,   519,   520,   521,   522,
     523,   524,   525,   526,   527,   531,   532,   533,   537,   537,
     541,   541,   545,   546,   550,   550,   554,   554,   558,   558,
     559,   559,   560,   560,   564,   565,   566,   570,   574,   578,
     582,   586,   590,   591,   592,   596,   597,   606,   607,   611,
     612,   616,   620,   624,   628,   632,   636,   637,   638,   639,
     640,   641,   642,   643,   644,   645,   646,   650,   651,   655,
     656,   660,   661,   662,   663,   664,   665,   669,   670,   674,
     675,   679,   680,   681,   682,   683,   684,   685,   686,   687,
     688,   689,   693,   694,   698,   699,   703,   707,   711,   712,
     716,   720,   721,   725,   729,   730,   734,   738,   739,   743,
     752,   753,   757,   761,   765,   769,   773,   777,   778,   779,
     780,   781,   782,   786,   787,   791,   795,   796,   800,   801,
     805,   806
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
  "GLOBAL", "PARALLEL", "NIL", "DOUBLE_BRACE_OPEN", "DOUBLE_BRACE_CLOSE",
  "NAME", "BOOL_LITERAL", "INT_LITERAL", "REAL_LITERAL", "STRING_LITERAL",
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
  "logical_or_operator", "logical_or_expression", "assign_operator",
  "assign_expression", "expression", "optional_expression",
  "expression_list", "local_variable", "span_expression", "span_list",
  "brackets", "parameters", "optional_parameters", "parameter_list",
  "parameter", "options", "option_list", "option", "arguments",
  "optional_arguments", "size", "generics", "generic_list", "generic",
  "optional_generics", "generic_arguments", "generic_argument_list",
  "generic_argument", "optional_generic_arguments",
  "global_variable_declaration", "local_variable_declaration",
  "member_variable_declaration", "function_declaration", "$@1",
  "fiber_declaration", "$@2", "program_declaration", "$@3",
  "member_function_declaration", "$@4", "member_fiber_declaration", "$@5",
  "binary_operator", "unary_operator", "binary_operator_declaration",
  "$@6", "unary_operator_declaration", "$@7", "assignment_operator",
  "assignment_operator_declaration", "$@8",
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
  "unknown_type", "tuple_type", "sequence_type", "postfix_type",
  "function_type", "type", "type_list", "parameter_type_list",
  "parameter_types", YY_NULLPTR
};
#endif

#define YYPACT_NINF -338
#define YYTABLE_NINF -211

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const short int yypact[] =
{
     290,    23,    23,    23,    23,    23,    30,    23,    18,    18,
    -338,    24,  -338,  -338,  -338,  -338,  -338,  -338,  -338,  -338,
    -338,  -338,  -338,   290,  -338,  -338,    64,    37,   100,    93,
     117,   117,    68,   111,   148,  -338,  -338,    62,  -338,  -338,
      13,  -338,    11,  -338,    -1,    23,  -338,    23,    25,   161,
     161,  -338,  -338,  -338,   123,   198,    23,    22,   144,  -338,
      62,    62,   159,   153,  -338,  -338,  -338,   142,  -338,    21,
    -338,   160,   169,   174,    40,  -338,  -338,   167,   178,    23,
    -338,   172,  -338,   171,   173,  -338,   196,   193,    62,  -338,
    -338,  -338,    62,  -338,  -338,  -338,  -338,  -338,  -338,  -338,
    -338,  -338,  -338,  -338,  -338,    23,   199,  -338,   187,   197,
     154,  -338,  -338,   207,   201,   202,   114,   161,  -338,   182,
     409,  -338,  -338,    28,   511,   443,  -338,   200,   203,    62,
    -338,    23,  -338,   155,  -338,  -338,  -338,    23,   111,  -338,
      23,    87,  -338,  -338,  -338,    23,  -338,    40,    40,  -338,
     214,   161,  -338,    62,   206,    62,  -338,  -338,  -338,   212,
     218,  -338,  -338,  -338,  -338,  -338,  -338,  -338,  -338,  -338,
    -338,  -338,   511,   477,   117,  -338,  -338,  -338,   217,    -2,
    -338,  -338,  -338,  -338,  -338,  -338,  -338,  -338,  -338,  -338,
    -338,   220,   222,  -338,  -338,   145,  -338,   511,  -338,    79,
     113,    55,   175,   237,   180,  -338,  -338,   225,   229,   230,
    -338,   221,  -338,  -338,   228,   234,  -338,  -338,   252,  -338,
     511,   511,   223,   511,   511,   511,  -338,    84,    16,   239,
    -338,  -338,  -338,  -338,  -338,   277,  -338,  -338,  -338,  -338,
    -338,  -338,   371,  -338,   226,  -338,  -338,   241,   259,  -338,
     119,  -338,  -338,  -338,  -338,  -338,   161,  -338,  -338,  -338,
      62,  -338,   260,  -338,   257,   161,   206,   263,    23,    23,
     511,  -338,    23,  -338,  -338,  -338,  -338,  -338,  -338,   511,
    -338,  -338,   511,  -338,  -338,  -338,  -338,   511,  -338,  -338,
     511,  -338,   511,  -338,  -338,  -338,   511,   511,   511,  -338,
    -338,  -338,   511,  -338,  -338,  -338,   223,   223,   293,   251,
    -338,   253,   261,    62,  -338,  -338,  -338,   511,  -338,     5,
    -338,  -338,  -338,  -338,  -338,    23,    23,   189,   264,  -338,
    -338,  -338,  -338,  -338,  -338,   119,  -338,   255,  -338,    40,
    -338,  -338,  -338,    40,  -338,   280,  -338,  -338,  -338,   278,
     283,   284,  -338,  -338,    79,   113,    55,   175,   237,  -338,
    -338,  -338,   320,  -338,   511,  -338,  -338,  -338,   142,   105,
     271,    23,   274,  -338,   321,    87,   117,   117,    23,  -338,
      62,  -338,  -338,    40,  -338,  -338,   511,   511,  -338,   511,
      20,   275,   259,  -338,  -338,  -338,   324,    62,   511,  -338,
     161,   161,  -338,    40,   142,    42,  -338,   294,  -338,  -338,
    -338,  -338,  -338,  -338,   511,  -338,   295,  -338,  -338,    40,
    -338,    49,  -338,   279,   281,  -338,   298,   511,    40,    40,
    -338,  -338,   282,  -338,  -338,   511,   223,  -338,  -338,  -338,
     299,  -338,   223,  -338
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const unsigned short int yydefact[] =
{
     235,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       2,     0,   221,   222,   223,   224,   225,   226,   227,   228,
     229,   230,   231,   232,   234,   236,     0,     0,   114,     0,
       0,     0,     0,     0,     0,   178,   179,     0,   233,     1,
       0,   137,     0,   113,    93,     0,   176,     0,     0,   239,
     239,   157,   155,   156,     0,     0,     0,     0,     0,   249,
       0,     0,     0,   121,   257,   258,   259,   263,   265,     0,
      97,     0,     0,    99,     0,   108,   112,     0,   110,     0,
      92,   170,   252,     0,     0,    90,     0,    94,     0,   238,
     133,   135,     0,   153,   154,   149,   150,   151,   152,   145,
     146,   143,   144,   147,   148,     0,     0,   115,     0,   117,
     263,   119,   177,   266,     0,     0,     0,   239,   120,   251,
       0,   261,   260,     0,     0,     0,   122,     0,     0,     0,
      98,     0,   245,   185,   244,   138,   109,     0,   121,   172,
       0,     0,   174,   175,    91,     0,   237,     0,     0,    96,
       0,   239,   116,     0,     0,     0,   255,   256,   270,   268,
       0,   264,   250,   254,    17,    18,    19,    20,     3,     4,
       5,     6,     0,     0,     0,    45,    43,    44,   106,    11,
       7,     8,     9,    10,    21,    22,    23,    24,    25,    26,
      27,     0,     0,    28,    34,    41,    46,     0,    50,    54,
      60,    64,    67,    70,    74,    76,    86,    87,     0,     0,
     125,     0,   240,   102,    79,     0,   123,   124,   242,   100,
       0,     0,     0,     0,    78,     0,   186,    11,     0,     0,
     196,   206,   198,   197,   199,     0,   200,   201,   202,   203,
     204,   205,   185,   209,     0,   111,   253,     0,   105,   248,
     220,   247,   171,    95,   134,   136,   239,   160,   118,   267,
       0,   271,     0,    14,     0,   239,     0,     0,     0,     0,
       0,    42,     0,    40,    38,    39,    47,    48,    49,     0,
      52,    53,     0,    58,    59,    56,    57,     0,    62,    63,
       0,    66,     0,    72,    73,    69,     0,     0,     0,    89,
     262,   126,     0,   103,   241,   101,     0,     0,     0,     0,
      77,     0,     0,     0,   162,   181,   163,     0,   127,     0,
     208,   243,   173,   104,   168,     0,     0,     0,     0,   211,
     212,   213,   214,   215,   216,   217,   219,     0,   158,     0,
     269,    12,    13,     0,   107,     0,    11,    35,    36,    31,
       0,    30,    37,    51,    55,    61,    65,    68,    71,    75,
      88,    80,   184,   191,     0,   193,   194,   195,   263,    81,
       0,     0,    11,   187,     0,     0,     0,     0,     0,   166,
       0,   218,   246,     0,   161,    16,     0,     0,    33,     0,
       0,     0,    84,    82,    83,   180,     0,     0,     0,   169,
     239,   239,   164,     0,   263,     0,   159,     0,    32,    29,
     183,   182,   192,    85,     0,   188,     0,   139,   141,     0,
     167,     0,   128,     0,     0,    15,     0,     0,     0,     0,
     165,   131,     0,   129,   130,     0,     0,   140,   142,   132,
       0,   190,     0,   189
};

  /* YYPGOTO[NTERM-NUM].  */
static const short int yypgoto[] =
{
    -338,     0,  -338,  -338,  -338,  -338,  -338,  -243,  -338,  -338,
    -338,  -338,  -338,  -338,  -338,  -338,  -338,  -338,   -42,  -338,
    -338,  -338,  -338,  -177,  -338,    65,  -338,    59,  -338,    58,
    -338,    57,  -338,    54,  -338,  -338,  -338,    63,  -106,  -338,
    -156,  -338,  -338,    61,  -337,   -22,  -338,   216,   -20,  -338,
     231,  -338,  -104,  -338,    86,  -338,   227,  -338,  -338,   330,
     213,  -338,   -39,  -338,  -338,  -338,  -338,  -338,  -338,  -338,
    -338,  -338,  -338,  -338,  -338,  -338,  -338,  -338,  -338,  -338,
    -338,  -338,  -338,  -338,  -338,  -338,  -338,  -338,  -338,  -338,
    -338,  -338,  -338,  -112,  -212,  -338,  -338,   -25,  -338,    -4,
    -338,  -338,  -338,  -338,  -338,  -338,  -338,   127,  -338,  -338,
      35,  -338,  -338,   349,  -338,  -338,    46,   -44,  -207,  -338,
    -209,  -113,  -338,     1,   365,  -338,   328,   242,  -338,  -338,
    -338,   -27,  -338,   -29,   -46,   124,  -338
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const short int yydefgoto[] =
{
      -1,   179,   180,   181,   182,   183,   184,   185,   186,   187,
     188,   189,   190,   191,   192,   193,   194,   349,   350,   274,
     195,   196,   197,   198,   279,   199,   282,   200,   287,   201,
     290,   202,   292,   203,   296,   204,   297,   205,   214,   311,
     215,   229,   207,   208,   123,    49,    81,    86,    87,    41,
      72,    73,   127,   324,   209,    43,    77,    78,    44,   118,
     108,   109,   267,    12,   230,   329,    13,   147,    14,   148,
      15,    74,   330,   428,   331,   429,   105,    56,    16,   383,
      17,   339,   317,   332,   419,   333,   403,    18,   375,   141,
     247,    19,    20,    21,    22,   232,   233,   234,   235,   374,
     236,   237,   238,   239,   240,   241,   242,   243,   244,   335,
     336,   337,    23,    24,    25,    26,    89,    90,   128,   305,
     134,   135,   251,   252,    35,   163,    83,   139,    64,    65,
      66,   110,    68,   113,   114,   160,   117
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const short int yytable[] =
{
      11,    27,    28,    29,    30,    31,    91,    33,    69,    50,
      67,   304,    55,   308,   206,   115,   262,   264,   212,   211,
     276,   231,    80,    11,   119,   347,   348,   228,   111,   352,
     220,   392,    54,    10,   254,   255,   106,    63,   334,    10,
      71,    10,    76,    48,    34,    82,   -11,    82,    54,   371,
      10,    10,   314,    10,   124,    57,    54,    63,    70,   146,
      63,    63,    79,   149,    39,   125,    60,   421,    61,    75,
      85,    62,   125,   161,    32,   124,   373,   315,   316,   138,
     107,    40,   126,    37,   133,   150,   125,   159,    63,   210,
      10,   275,    63,   125,   283,   284,    10,   362,   363,   246,
     218,   132,   353,   422,   133,    54,    60,   257,    61,   259,
     431,    62,   285,   286,   306,   307,    63,   309,   310,   312,
      51,    52,    53,   334,   111,   325,   326,   327,   373,    63,
     231,    71,   -11,   227,   277,   278,   228,    76,   124,     9,
     138,    57,    10,   313,   323,    54,   361,    10,   249,   125,
      45,   250,   265,    63,    46,    63,    47,    42,    60,   158,
      61,    48,   394,    62,   351,   220,   280,   281,    57,   221,
     222,   223,   224,   225,     8,    59,   164,   165,   166,   226,
     167,   411,    92,    10,   168,   169,   170,   171,   120,   125,
     121,   270,   206,   271,   122,    88,   272,   273,   424,   172,
     154,   173,   121,   116,   174,   112,   122,   175,   176,   177,
      57,   370,   338,   293,   130,   294,   288,   289,   295,   129,
    -210,   343,   378,    88,   131,   136,   384,   441,   137,   140,
     385,   159,   142,   443,   143,    93,    94,    95,    96,    97,
      98,   144,   227,   145,   151,   152,   156,   153,   162,   157,
     328,    99,   100,   101,   102,   103,   104,   155,   391,   256,
      63,   216,   260,   261,   217,   393,   178,   266,   346,   346,
     406,   268,   346,   269,   291,   298,   299,   300,   302,   303,
     407,   351,   301,   409,   369,   124,   368,   133,   413,   319,
     420,   321,   416,     1,     2,     3,     4,     5,     6,     7,
     318,   423,   322,   125,   342,   341,   430,   364,   426,     8,
       9,   345,   365,    63,   366,   437,   438,   432,    10,   372,
     382,   436,   367,   380,   386,   376,   377,   389,   387,   440,
     388,   390,   395,   397,   398,   328,   412,   414,   427,   425,
     433,   435,   434,   439,   442,   408,   355,   354,   356,   357,
     358,   405,   344,   404,   400,   401,   417,   418,   402,   360,
     359,   253,   219,    58,   245,   410,   258,   396,   415,   320,
     381,   372,    38,   379,    36,    84,   399,     0,    54,     0,
      63,   220,   248,     0,   340,   221,   222,   223,   224,   225,
       8,     0,   164,   165,   166,   226,   167,    63,     0,    10,
     168,   169,   170,   171,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   172,     0,   173,     0,     0,
     174,     0,     0,   175,   176,   177,     0,     0,     0,     0,
     164,   165,   166,     0,   167,     0,  -207,    10,   168,   169,
     170,   171,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   172,     0,   173,     0,     0,   174,     0,
       0,   175,   176,   177,   164,   165,   166,     0,   167,   178,
       0,    10,   168,   169,   170,   171,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   172,   213,   173,
       0,     0,   174,     0,     0,   175,   176,   177,   164,   165,
     166,     0,   167,     0,     0,    10,   168,   169,   170,   171,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   172,     0,   173,   263,     0,   174,     0,     0,   175,
     176,   177,   164,   165,   166,     0,   167,     0,     0,    10,
     168,   169,   170,   171,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   172,     0,   173,     0,     0,
     174,     0,     0,   175,   176,   177
};

static const short int yycheck[] =
{
       0,     1,     2,     3,     4,     5,    50,     7,    37,    31,
      37,   218,    32,   222,   120,    61,   172,   173,   124,   123,
     197,   133,    44,    23,    63,   268,   269,   133,    57,   272,
      10,   368,    32,    28,   147,   148,    56,    37,   250,    28,
      40,    28,    42,    44,    26,    45,    48,    47,    48,    44,
      28,    28,    36,    28,    33,    57,    56,    57,    45,    88,
      60,    61,    63,    92,     0,    44,    44,   404,    46,    58,
      45,    49,    44,   117,    44,    33,   319,    61,    62,    79,
      58,    44,    61,    59,    64,   105,    44,   116,    88,    61,
      28,   195,    92,    44,    39,    40,    28,   306,   307,   138,
     129,    61,   279,    61,    64,   105,    44,   151,    46,   155,
      61,    49,    57,    58,   220,   221,   116,   223,   224,   225,
      52,    53,    54,   335,   153,     6,     7,     8,   371,   129,
     242,   131,    48,   133,    55,    56,   242,   137,    33,    20,
     140,    57,    28,    59,   248,   145,   302,    28,    61,    44,
      57,    64,   174,   153,    61,   155,    63,    57,    44,    45,
      46,    44,   369,    49,   270,    10,    53,    54,    57,    14,
      15,    16,    17,    18,    19,    27,    21,    22,    23,    24,
      25,   390,    59,    28,    29,    30,    31,    32,    46,    44,
      48,    46,   298,    48,    52,    34,    51,    52,   405,    44,
      46,    46,    48,    44,    49,    61,    52,    52,    53,    54,
      57,   317,   256,    33,    45,    35,    41,    42,    38,    59,
      65,   265,    33,    34,    50,    58,   339,   436,    50,    57,
     343,   260,    61,   442,    61,    37,    38,    39,    40,    41,
      42,    45,   242,    50,    45,    58,    45,    50,    66,    47,
     250,    53,    54,    55,    56,    57,    58,    50,   364,    45,
     260,    61,    50,    45,    61,   369,    60,    50,   268,   269,
     383,    51,   272,    51,    37,    50,    47,    47,    50,    45,
     386,   387,    61,   389,   313,    33,   313,    64,   392,    12,
     403,    65,   398,     3,     4,     5,     6,     7,     8,     9,
      61,   405,    61,    44,    47,    45,   419,    14,   414,    19,
      20,    48,    61,   313,    61,   428,   429,   421,    28,   319,
      65,   427,    61,    59,    44,   325,   326,    43,    50,   435,
      47,    11,    61,    59,    13,   335,    61,    13,    43,    45,
      61,    43,    61,    61,    45,   387,   287,   282,   290,   292,
     296,   380,   266,   380,   376,   377,   400,   401,   378,   298,
     297,   145,   131,    33,   137,   390,   153,   371,   397,   242,
     335,   371,    23,   327,     9,    47,   375,    -1,   378,    -1,
     380,    10,   140,    -1,   260,    14,    15,    16,    17,    18,
      19,    -1,    21,    22,    23,    24,    25,   397,    -1,    28,
      29,    30,    31,    32,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    44,    -1,    46,    -1,    -1,
      49,    -1,    -1,    52,    53,    54,    -1,    -1,    -1,    -1,
      21,    22,    23,    -1,    25,    -1,    65,    28,    29,    30,
      31,    32,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    44,    -1,    46,    -1,    -1,    49,    -1,
      -1,    52,    53,    54,    21,    22,    23,    -1,    25,    60,
      -1,    28,    29,    30,    31,    32,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    44,    45,    46,
      -1,    -1,    49,    -1,    -1,    52,    53,    54,    21,    22,
      23,    -1,    25,    -1,    -1,    28,    29,    30,    31,    32,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    44,    -1,    46,    47,    -1,    49,    -1,    -1,    52,
      53,    54,    21,    22,    23,    -1,    25,    -1,    -1,    28,
      29,    30,    31,    32,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    44,    -1,    46,    -1,    -1,
      49,    -1,    -1,    52,    53,    54
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,     3,     4,     5,     6,     7,     8,     9,    19,    20,
      28,    68,   130,   133,   135,   137,   145,   147,   154,   158,
     159,   160,   161,   179,   180,   181,   182,    68,    68,    68,
      68,    68,    44,    68,    26,   191,   191,    59,   180,     0,
      44,   116,    57,   122,   125,    57,    61,    63,    44,   112,
     112,    52,    53,    54,    68,   115,   144,    57,   126,    27,
      44,    46,    49,    68,   195,   196,   197,   198,   199,   200,
      45,    68,   117,   118,   138,    58,    68,   123,   124,    63,
     112,   113,    68,   193,   193,    45,   114,   115,    34,   183,
     184,   184,    59,    37,    38,    39,    40,    41,    42,    53,
      54,    55,    56,    57,    58,   143,   115,    58,   127,   128,
     198,   200,    61,   200,   201,   201,    44,   203,   126,   129,
      46,    48,    52,   111,    33,    44,    61,   119,   185,    59,
      45,    50,    61,    64,   187,   188,    58,    50,    68,   194,
      57,   156,    61,    61,    45,    50,   200,   134,   136,   200,
     115,    45,    58,    50,    46,    50,    45,    47,    45,   200,
     202,   184,    66,   192,    21,    22,    23,    25,    29,    30,
      31,    32,    44,    46,    49,    52,    53,    54,    60,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    87,    88,    89,    90,    92,
      94,    96,    98,   100,   102,   104,   105,   109,   110,   121,
      61,   119,   105,    45,   105,   107,    61,    61,   200,   117,
      10,    14,    15,    16,    17,    18,    24,    68,   105,   108,
     131,   160,   162,   163,   164,   165,   167,   168,   169,   170,
     171,   172,   173,   174,   175,   123,   129,   157,   194,    61,
      64,   189,   190,   114,   188,   188,    45,   184,   127,   201,
      50,    45,   107,    47,   107,   112,    50,   129,    51,    51,
      46,    48,    51,    52,    86,   119,    90,    55,    56,    91,
      53,    54,    93,    39,    40,    57,    58,    95,    41,    42,
      97,    37,    99,    33,    35,    38,   101,   103,    50,    47,
      47,    61,    50,    45,   185,   186,   105,   105,   187,   105,
     105,   106,   105,    59,    36,    61,    62,   149,    61,    12,
     174,    65,    61,   119,   120,     6,     7,     8,    68,   132,
     139,   141,   150,   152,   161,   176,   177,   178,   184,   148,
     202,    45,    47,   184,   121,    48,    68,    74,    74,    84,
      85,   105,    74,    90,    92,    94,    96,    98,   100,   104,
     110,   107,   187,   187,    14,    61,    61,    61,   198,   200,
     105,    44,    68,    74,   166,   155,    68,    68,    33,   183,
      59,   177,    65,   146,   188,   188,    44,    50,    47,    43,
      11,   105,   111,   119,   185,    61,   166,    59,    13,   190,
     112,   112,   115,   153,   198,   200,   188,   105,    85,   105,
     164,   187,    61,   119,    13,   200,   105,   184,   184,   151,
     188,   111,    61,   119,   185,    45,   105,    43,   140,   142,
     188,    61,   119,    61,    61,    43,   105,   188,   188,    61,
     105,   187,    45,   187
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,    67,    68,    69,    70,    71,    72,    73,    73,    73,
      73,    74,    75,    76,    76,    77,    78,    79,    80,    81,
      82,    83,    83,    83,    83,    83,    83,    83,    83,    84,
      84,    85,    85,    86,    87,    87,    87,    87,    87,    87,
      87,    88,    88,    89,    89,    89,    90,    90,    91,    91,
      92,    92,    93,    93,    94,    94,    95,    95,    95,    95,
      96,    96,    97,    97,    98,    98,    99,   100,   100,   101,
     102,   102,   103,   103,   104,   104,   105,   106,   106,   107,
     107,   108,   108,   108,   108,   108,   109,   110,   110,   111,
     112,   112,   113,   113,   114,   114,   115,   116,   116,   117,
     117,   118,   119,   119,   120,   120,   121,   121,   122,   122,
     123,   123,   124,   125,   125,   126,   126,   127,   127,   128,
     129,   129,   130,   130,   130,   130,   130,   131,   132,   132,
     132,   132,   132,   134,   133,   136,   135,   138,   137,   140,
     139,   142,   141,   143,   143,   143,   143,   143,   143,   143,
     143,   143,   143,   143,   143,   144,   144,   144,   146,   145,
     148,   147,   149,   149,   151,   150,   153,   152,   155,   154,
     156,   154,   157,   154,   158,   158,   158,   159,   160,   161,
     162,   163,   164,   164,   164,   165,   165,   166,   166,   167,
     167,   168,   169,   170,   171,   172,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   173,   173,   174,   174,   175,
     175,   176,   176,   176,   176,   176,   176,   177,   177,   178,
     178,   179,   179,   179,   179,   179,   179,   179,   179,   179,
     179,   179,   180,   180,   181,   181,   182,   183,   184,   184,
     185,   186,   186,   187,   188,   188,   189,   190,   190,   191,
     192,   192,   193,   194,   195,   196,   197,   198,   198,   198,
     198,   198,   198,   199,   199,   200,   201,   201,   202,   202,
     203,   203
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
       3,     3,     4,     4,     4,     5,     1,     1,     3,     3,
       2,     3,     1,     0,     1,     3,     3,     2,     3,     1,
       3,     4,     2,     3,     1,     0,     1,     3,     2,     3,
       1,     3,     1,     1,     0,     2,     3,     1,     3,     1,
       1,     0,     4,     5,     5,     5,     6,     2,     4,     5,
       5,     5,     6,     0,     6,     0,     6,     0,     5,     0,
       6,     0,     6,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     0,     9,
       0,     8,     1,     1,     0,     5,     0,     4,     0,     9,
       0,     6,     0,     7,     5,     5,     3,     4,     2,     2,
       4,     2,     5,     5,     3,     0,     1,     1,     3,    10,
       8,     3,     5,     3,     3,     3,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     2,     1,
       0,     1,     1,     1,     1,     1,     1,     1,     2,     1,
       0,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     2,     1,     0,     1,     2,     1,     0,
       2,     1,     0,     3,     1,     1,     3,     1,     1,     2,
       1,     0,     1,     2,     3,     3,     3,     1,     1,     1,
       2,     2,     4,     1,     3,     1,     1,     3,     1,     3,
       2,     3
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
       0,     0
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
       0,     0
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
       0,     0
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
       0,     0,     0,     0,     0,     0,     5,     0,     0,     0,
       0,     0,     0,     0,     0,     7,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     9,     0,     0,     0,     0,     0,     0,     0,
       0,    11,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     1,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       3,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
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

/* YYCONFL[I] -- lists of conflicting rule numbers, each terminated by
   0, pointed into by YYCONFLP.  */
static const short int yyconfl[] =
{
       0,   114,     0,   121,     0,   121,     0,    11,     0,   121,
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
#line 1458 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 3:
#line 169 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Literal<bool>((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valBool), yytext, new bi::BasicType(new bi::Name("Boolean")), make_loc((*yylocp))); }
#line 1464 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 4:
#line 173 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Literal<int64_t>((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valInt), yytext, new bi::BasicType(new bi::Name("Integer")), make_loc((*yylocp))); }
#line 1470 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 5:
#line 177 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Literal<double>((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valReal), yytext, new bi::BasicType(new bi::Name("Real")), make_loc((*yylocp))); }
#line 1476 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 6:
#line 181 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Literal<const char*>((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valString), yytext, new bi::BasicType(new bi::Name("String")), make_loc((*yylocp))); }
#line 1482 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 11:
#line 192 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Identifier<bi::Unknown>((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valName), make_loc((*yylocp))); }
#line 1488 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 12:
#line 196 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Parentheses((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1494 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 13:
#line 200 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Sequence((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1500 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 14:
#line 201 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Sequence(empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1506 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 15:
#line 205 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Cast(new bi::UnknownType(false, (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1512 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 16:
#line 209 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LambdaFunction((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 1518 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 17:
#line 213 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::This(make_loc((*yylocp))); }
#line 1524 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 18:
#line 217 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Super(make_loc((*yylocp))); }
#line 1530 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 19:
#line 221 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Global(make_loc((*yylocp))); }
#line 1536 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 20:
#line 225 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Nil(make_loc((*yylocp))); }
#line 1542 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 29:
#line 240 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Range((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1548 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 30:
#line 241 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Index((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1554 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 32:
#line 246 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1560 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 33:
#line 250 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1566 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 35:
#line 255 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Member((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1572 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 36:
#line 256 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Member((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1578 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 37:
#line 257 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Member((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1584 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 38:
#line 258 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Slice((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1590 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 39:
#line 259 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Call((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1596 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 40:
#line 260 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Get((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1602 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 42:
#line 265 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Query((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1608 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 43:
#line 269 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::UnaryOperator>(new bi::Name("+"), make_loc((*yylocp))); }
#line 1614 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 44:
#line 270 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::UnaryOperator>(new bi::Name("-"), make_loc((*yylocp))); }
#line 1620 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 45:
#line 271 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::UnaryOperator>(new bi::Name("!"), make_loc((*yylocp))); }
#line 1626 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 47:
#line 276 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::UnaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1632 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 48:
#line 280 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("*"), make_loc((*yylocp))); }
#line 1638 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 49:
#line 281 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("/"), make_loc((*yylocp))); }
#line 1644 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 51:
#line 286 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), make_loc((*yylocp))); }
#line 1650 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 52:
#line 290 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("+"), make_loc((*yylocp))); }
#line 1656 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 53:
#line 291 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("-"), make_loc((*yylocp))); }
#line 1662 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 55:
#line 296 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), make_loc((*yylocp))); }
#line 1668 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 56:
#line 300 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("<"), make_loc((*yylocp))); }
#line 1674 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 57:
#line 301 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name(">"), make_loc((*yylocp))); }
#line 1680 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 58:
#line 302 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("<="), make_loc((*yylocp))); }
#line 1686 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 59:
#line 303 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name(">="), make_loc((*yylocp))); }
#line 1692 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 61:
#line 308 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), make_loc((*yylocp))); }
#line 1698 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 62:
#line 312 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("=="), make_loc((*yylocp))); }
#line 1704 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 63:
#line 313 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("!="), make_loc((*yylocp))); }
#line 1710 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 65:
#line 318 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), make_loc((*yylocp))); }
#line 1716 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 66:
#line 322 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("&&"), make_loc((*yylocp))); }
#line 1722 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 68:
#line 327 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), make_loc((*yylocp))); }
#line 1728 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 69:
#line 331 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::OverloadedIdentifier<bi::BinaryOperator>(new bi::Name("||"), make_loc((*yylocp))); }
#line 1734 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 71:
#line 336 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::BinaryCall((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), make_loc((*yylocp))); }
#line 1740 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 72:
#line 340 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("<-"); }
#line 1746 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 73:
#line 341 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("<~"); }
#line 1752 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 75:
#line 346 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Assign((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1758 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 78:
#line 355 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1764 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 80:
#line 360 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1770 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 81:
#line 364 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LocalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1776 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 82:
#line 365 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LocalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1782 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 83:
#line 366 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LocalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1788 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 84:
#line 367 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LocalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1794 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 85:
#line 368 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LocalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1800 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 86:
#line 372 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Span((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1806 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 88:
#line 377 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1812 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 89:
#line 381 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1818 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 90:
#line 385 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1824 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 91:
#line 386 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1830 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 93:
#line 391 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1836 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 95:
#line 396 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1842 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 96:
#line 400 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Parameter((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1848 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 97:
#line 404 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1854 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 98:
#line 405 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1860 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 100:
#line 410 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1866 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 101:
#line 414 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Parameter((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1872 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 102:
#line 418 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1878 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 103:
#line 419 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1884 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 105:
#line 424 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1890 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 106:
#line 428 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valInt) = 1; }
#line 1896 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 107:
#line 429 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valInt) = (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valInt) + 1; }
#line 1902 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 108:
#line 433 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1908 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 109:
#line 434 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1914 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 111:
#line 439 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::ExpressionList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1920 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 112:
#line 443 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::Generic((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valName), empty_type((*yylocp)), make_loc((*yylocp))); }
#line 1926 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 114:
#line 448 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1932 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 115:
#line 452 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 1938 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 116:
#line 453 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType); }
#line 1944 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 118:
#line 458 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::TypeList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 1950 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 121:
#line 467 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 1956 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 122:
#line 476 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1962 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 123:
#line 477 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1968 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 124:
#line 478 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 1974 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 125:
#line 479 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1980 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 126:
#line 480 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1986 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 127:
#line 484 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::ExpressionStatement((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 1992 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 128:
#line 488 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 1998 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 129:
#line 489 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2004 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 130:
#line 490 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 2010 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 131:
#line 491 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2016 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 132:
#line 492 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2022 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 133:
#line 496 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2028 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 134:
#line 496 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Function(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2034 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 135:
#line 500 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2040 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 136:
#line 500 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Fiber(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), new bi::FiberType((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2046 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 137:
#line 504 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2052 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 138:
#line 504 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Program((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2058 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 139:
#line 508 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2064 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 140:
#line 508 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::MemberFunction(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2070 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 141:
#line 512 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2076 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 142:
#line 512 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::MemberFiber(bi::NONE, (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), new bi::FiberType((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2082 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 143:
#line 516 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('*'); }
#line 2088 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 144:
#line 517 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('/'); }
#line 2094 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 145:
#line 518 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('+'); }
#line 2100 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 146:
#line 519 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('-'); }
#line 2106 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 147:
#line 520 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('<'); }
#line 2112 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 148:
#line 521 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('>'); }
#line 2118 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 149:
#line 522 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("<="); }
#line 2124 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 150:
#line 523 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name(">="); }
#line 2130 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 151:
#line 524 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("=="); }
#line 2136 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 152:
#line 525 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("!="); }
#line 2142 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 153:
#line 526 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("&&"); }
#line 2148 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 154:
#line 527 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("||"); }
#line 2154 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 155:
#line 531 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('+'); }
#line 2160 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 156:
#line 532 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('-'); }
#line 2166 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 157:
#line 533 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name('!'); }
#line 2172 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 158:
#line 537 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2178 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 159:
#line 537 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::BinaryOperator((((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), new bi::Binary((((yyGLRStackItem const *)yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2184 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 160:
#line 541 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2190 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 161:
#line 541 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::UnaryOperator((((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2196 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 162:
#line 545 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("~>"); }
#line 2202 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 163:
#line 546 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valName) = new bi::Name("~"); }
#line 2208 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 164:
#line 550 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2214 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 165:
#line 550 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::AssignmentOperator((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2220 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 166:
#line 554 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2226 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 167:
#line 554 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::ConversionOperator((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2232 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 168:
#line 558 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2238 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 169:
#line 558 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Class((((yyGLRStackItem const *)yyvsp)[YYFILL (-7)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), false, (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2244 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 170:
#line 559 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2250 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 171:
#line 559 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Class((((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), empty_type((*yylocp)), false, empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2256 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 172:
#line 560 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); }
#line 2262 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 173:
#line 560 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Class((((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), true, empty_expr((*yylocp)), empty_stmt((*yylocp)), make_doc_loc((*yylocp))); }
#line 2268 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 174:
#line 564 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Basic((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), false, make_doc_loc((*yylocp))); }
#line 2274 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 175:
#line 565 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Basic((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), true, make_doc_loc((*yylocp))); }
#line 2280 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 176:
#line 566 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Basic((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), empty_type((*yylocp)), false, make_doc_loc((*yylocp))); }
#line 2286 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 177:
#line 570 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Explicit(new bi::ClassType((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))), make_doc_loc((*yylocp))); }
#line 2292 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 178:
#line 574 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Raw(new bi::Name("cpp"), pop_raw(), make_loc((*yylocp))); }
#line 2298 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 179:
#line 578 "bi/parser.ypp" /* glr.c:816  */
    { push_raw(); ((*yyvalp).valStatement) = new bi::Raw(new bi::Name("hpp"), pop_raw(), make_loc((*yylocp))); }
#line 2304 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 180:
#line 582 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Assignment((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2310 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 181:
#line 586 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::ExpressionStatement((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2316 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 182:
#line 590 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::If((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2322 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 183:
#line 591 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::If((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2328 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 184:
#line 592 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::If((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), empty_stmt((*yylocp)), make_loc((*yylocp))); }
#line 2334 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 185:
#line 596 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valAnnotation) = bi::NONE; }
#line 2340 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 186:
#line 597 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valAnnotation) = bi::PARALLEL; }
#line 2346 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 188:
#line 607 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = new bi::LocalVariable((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 2352 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 189:
#line 611 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::For((((yyGLRStackItem const *)yyvsp)[YYFILL (-9)].yystate.yysemantics.yysval.valAnnotation), (((yyGLRStackItem const *)yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2358 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 190:
#line 612 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::For((((yyGLRStackItem const *)yyvsp)[YYFILL (-7)].yystate.yysemantics.yysval.valAnnotation), (((yyGLRStackItem const *)yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2364 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 191:
#line 616 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::While((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2370 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 192:
#line 620 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::DoWhile((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2376 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 193:
#line 624 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Assert((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2382 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 194:
#line 628 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Return((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2388 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 195:
#line 632 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Yield((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2394 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 208:
#line 651 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::StatementList((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2400 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 210:
#line 656 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2406 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 218:
#line 670 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::StatementList((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2412 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 220:
#line 675 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2418 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 233:
#line 694 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::StatementList((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2424 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 235:
#line 699 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2430 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 236:
#line 703 "bi/parser.ypp" /* glr.c:816  */
    { compiler->setRoot((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement)); }
#line 2436 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 237:
#line 707 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType); }
#line 2442 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 239:
#line 712 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 2448 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 240:
#line 716 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression); }
#line 2454 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 242:
#line 721 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 2460 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 243:
#line 725 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Braces((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2466 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 245:
#line 730 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2472 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 246:
#line 734 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = new bi::Braces((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2478 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 248:
#line 739 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2484 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 250:
#line 752 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valBool) = true; }
#line 2490 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 251:
#line 753 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valBool) = false; }
#line 2496 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 252:
#line 757 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::BasicType((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valName), make_loc((*yylocp))); }
#line 2502 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 253:
#line 761 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::ClassType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2508 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 254:
#line 765 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::UnknownType((((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valBool), (((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2514 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 255:
#line 769 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::TupleType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2520 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 256:
#line 773 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::SequenceType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2526 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 260:
#line 780 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::FiberType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2532 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 261:
#line 781 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::OptionalType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2538 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 262:
#line 782 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::ArrayType((((yyGLRStackItem const *)yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valInt), make_loc((*yylocp))); }
#line 2544 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 264:
#line 787 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::FunctionType((((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2550 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 267:
#line 796 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::TypeList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2556 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 269:
#line 801 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = new bi::TypeList((((yyGLRStackItem const *)yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (((yyGLRStackItem const *)yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2562 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 270:
#line 805 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 2568 "bi/parser.cpp" /* glr.c:816  */
    break;

  case 271:
#line 806 "bi/parser.ypp" /* glr.c:816  */
    { ((*yyvalp).valType) = (((yyGLRStackItem const *)yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType); }
#line 2574 "bi/parser.cpp" /* glr.c:816  */
    break;


#line 2578 "bi/parser.cpp" /* glr.c:816  */
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
  (!!((Yystate) == (-338)))

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



#line 809 "bi/parser.ypp" /* glr.c:2584  */

