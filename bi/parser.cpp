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
#define YYLAST   580

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  73
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  134
/* YYNRULES -- Number of rules.  */
#define YYNRULES  268
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  461
/* YYMAXRHS -- Maximum number of symbols on right-hand side of rule.  */
#define YYMAXRHS 10
/* YYMAXLEFT -- Maximum number of symbols to the left of a handle
   accessed by $0, $-1, etc., in any rule.  */
#define YYMAXLEFT 0

/* YYMAXUTOK -- Last valid token number (for yychar).  */
#define YYMAXUTOK   304
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
       2,     2,     2,    58,     2,     2,     2,     2,    72,     2,
      50,    51,    61,    59,    56,    60,    57,    62,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    65,    67,
      63,    68,    64,    54,    55,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    52,     2,    53,     2,    66,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    70,     2,    71,    69,     2,     2,     2,
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
      45,    46,    47,    48,    49
};

#if YYDEBUG
/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   161,   161,   170,   174,   178,   182,   186,   187,   188,
     189,   193,   197,   201,   205,   209,   213,   217,   221,   225,
     226,   227,   228,   229,   230,   231,   232,   236,   237,   241,
     242,   246,   250,   251,   252,   253,   254,   255,   256,   257,
     261,   262,   266,   267,   268,   272,   273,   277,   278,   282,
     283,   287,   288,   292,   293,   297,   298,   299,   300,   309,
     310,   314,   315,   319,   320,   324,   328,   329,   333,   337,
     338,   342,   346,   347,   351,   355,   356,   360,   361,   365,
     369,   370,   374,   378,   379,   383,   384,   388,   389,   393,
     397,   398,   402,   403,   407,   411,   412,   416,   417,   421,
     422,   426,   427,   431,   432,   436,   440,   441,   445,   446,
     450,   451,   455,   459,   460,   469,   470,   471,   472,   473,
     477,   478,   479,   480,   481,   485,   486,   487,   488,   489,
     490,   494,   494,   498,   498,   502,   503,   504,   505,   511,
     511,   512,   512,   516,   516,   517,   517,   521,   521,   525,
     526,   527,   528,   529,   530,   534,   538,   538,   542,   542,
     546,   546,   550,   550,   554,   555,   556,   560,   560,   561,
     561,   562,   562,   566,   567,   568,   572,   576,   580,   581,
     582,   583,   587,   591,   595,   596,   597,   601,   605,   609,
     615,   616,   620,   624,   628,   632,   636,   640,   644,   645,
     646,   647,   648,   649,   650,   651,   652,   653,   654,   655,
     656,   660,   661,   665,   666,   670,   671,   672,   673,   674,
     675,   679,   680,   684,   685,   689,   690,   691,   692,   693,
     694,   695,   696,   697,   698,   702,   703,   707,   708,   712,
     716,   720,   721,   725,   729,   730,   734,   738,   739,   743,
     747,   748,   752,   761,   762,   766,   770,   771,   775,   776,
     777,   778,   779,   780,   781,   785,   786,   790,   791
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
  "LE_OP", "GE_OP", "EQ_OP", "NE_OP", "RANGE_OP", "SPIN_OP", "'('", "')'",
  "'['", "']'", "'?'", "'@'", "','", "'.'", "'!'", "'+'", "'-'", "'*'",
  "'/'", "'<'", "'>'", "':'", "'_'", "';'", "'='", "'~'", "'{'", "'}'",
  "'&'", "$accept", "name", "bool_literal", "int_literal", "real_literal",
  "string_literal", "literal", "identifier", "parens_expression",
  "sequence_expression", "cast_expression", "function_expression",
  "this_expression", "super_expression", "nil_expression",
  "primary_expression", "index_expression", "index_list", "slice",
  "postfix_expression", "query_expression", "prefix_operator",
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
  "optional_return_type", "value", "optional_value", "braces",
  "optional_braces", "class_braces", "optional_class_braces",
  "double_braces", "weak_modifier", "named_type", "primary_type", "type",
  "type_list", "optional_type_list", YY_NULLPTR
};
#endif

#define YYPACT_NINF (-329)
#define YYTABLE_NINF (-239)

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     161,    -4,    -4,    -4,    -4,   -10,     3,     3,  -329,  -329,
    -329,     7,  -329,  -329,  -329,  -329,  -329,  -329,    94,  -329,
    -329,  -329,  -329,   191,  -329,  -329,   105,    63,    53,    59,
      59,    10,   107,  -329,  -329,    -5,    -4,  -329,  -329,     2,
    -329,    -4,  -329,    -4,    -2,  -329,    93,    93,  -329,  -329,
    -329,   100,  -329,   516,    -4,  -329,    -5,   109,   113,   125,
    -329,    60,   127,  -329,   139,   155,   159,    75,   163,   168,
    -329,  -329,   174,   170,    25,   207,   207,    -5,  -329,  -329,
    -329,  -329,  -329,  -329,  -329,  -329,  -329,  -329,  -329,  -329,
    -329,  -329,  -329,  -329,  -329,  -329,    -4,   199,   160,   200,
      -5,    -3,  -329,   183,    -4,   514,   474,   434,  -329,    17,
    -329,    13,   190,   193,     5,    -5,  -329,    -4,  -329,   394,
    -329,  -329,  -329,  -329,  -329,    -4,  -329,   210,   206,    -5,
    -329,  -329,  -329,    41,   212,   207,   198,    -5,  -329,  -329,
     214,  -329,   203,   213,    41,  -329,  -329,  -329,  -329,  -329,
     211,  -329,  -329,  -329,  -329,  -329,   514,   514,    93,   208,
    -329,  -329,  -329,  -329,  -329,  -329,  -329,  -329,  -329,  -329,
    -329,   215,  -329,  -329,   175,  -329,   514,  -329,    22,    71,
     158,   162,   231,   -11,  -329,  -329,  -329,   218,   227,   224,
    -329,   226,   235,   236,  -329,  -329,  -329,   223,  -329,  -329,
      -4,  -329,   234,   104,  -329,    -4,   514,    -4,   514,   228,
     514,   514,   514,   288,  -329,   120,    67,  -329,  -329,  -329,
    -329,  -329,  -329,   277,  -329,  -329,  -329,  -329,  -329,  -329,
    -329,   394,  -329,   232,  -329,  -329,  -329,    -4,   182,    75,
      75,   207,  -329,  -329,   207,  -329,    -5,    -4,   251,   255,
     207,   250,    -4,  -329,   514,  -329,    -4,  -329,  -329,  -329,
    -329,   514,   514,   514,   514,   514,  -329,   514,   514,   514,
    -329,   198,   514,  -329,  -329,  -329,  -329,    -4,    90,  -329,
    -329,   272,   228,  -329,   297,   228,   298,   246,  -329,   248,
     249,    -4,    -5,  -329,  -329,  -329,  -329,  -329,   514,   305,
    -329,  -329,  -329,  -329,  -329,  -329,    75,  -329,  -329,   256,
    -329,  -329,  -329,    75,   268,  -329,   264,   269,   275,  -329,
    -329,    22,    71,   158,   162,   231,  -329,  -329,  -329,  -329,
     257,   276,  -329,   279,  -329,  -329,   258,   316,   514,  -329,
     514,  -329,  -329,  -329,   315,    96,   263,    -4,    75,  -329,
    -329,  -329,   514,   514,  -329,   514,  -329,  -329,  -329,    -4,
      -4,   204,  -329,  -329,   307,   267,  -329,   242,  -329,  -329,
    -329,  -329,  -329,   279,  -329,   265,  -329,    15,   287,   270,
     514,  -329,    21,   271,   273,  -329,   326,  -329,   291,  -329,
    -329,    90,    59,    59,    -4,  -329,  -329,    -5,    -4,    -4,
    -329,  -329,  -329,  -329,   514,  -329,   296,  -329,   281,  -329,
    -329,   514,  -329,  -329,    93,    93,  -329,    75,   134,    59,
      59,   228,   514,  -329,   301,   207,   207,    75,  -329,  -329,
      44,   283,   284,    93,    93,  -329,   228,   514,  -329,  -329,
    -329,  -329,   286,  -329,  -329,   207,   207,  -329,   228,    75,
      75,  -329,  -329,  -329,  -329,  -329,  -329,    75,    75,  -329,
    -329
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_int16 yydefact[] =
{
     166,     0,     0,     0,     0,     0,     0,     0,   165,   164,
       2,     0,   225,   226,   227,   228,   229,   230,     0,   231,
     232,   233,   234,   166,   237,   239,     0,     0,     0,   107,
     107,     0,     0,   176,   177,     0,     0,   236,     1,     0,
     147,     0,   175,     0,     0,   106,     0,     0,    44,    42,
      43,     0,   155,     0,     0,   252,     0,     0,   114,   256,
     258,     0,   107,    90,     0,     0,    92,     0,     0,     0,
     101,   105,     0,   103,     0,   242,   242,     0,    65,    68,
      57,    58,    61,    62,    51,    52,    47,    48,    55,    56,
     149,   150,   151,   152,   153,   154,     0,     0,   265,     0,
     268,     0,   113,   254,     0,     0,     0,     0,   259,   260,
     115,     0,     0,     0,    86,     0,    91,     0,   248,   214,
     247,   148,   173,   174,   102,     0,    83,     0,    87,     0,
     241,   131,   133,    89,     0,   242,     0,     0,   257,   267,
       0,   108,     0,   110,   112,   253,   255,   262,    16,    17,
       0,    18,     3,     4,     5,     6,     0,     0,     0,   114,
       7,     8,     9,    10,    19,    20,    21,    22,    23,    24,
      25,     0,    26,    32,    40,    45,     0,    49,    53,    59,
      63,    66,    69,    72,    74,   243,    95,    77,     0,    99,
      79,    80,     0,     0,   256,   261,   118,     0,   116,   117,
       0,    85,   169,   245,    93,     0,     0,     0,     0,     0,
       0,    76,     0,     0,   189,   114,     0,   198,   210,   200,
     199,   201,   202,     0,   203,   204,   205,   206,   207,   208,
     209,   211,   213,     0,   194,   104,    84,     0,   240,     0,
       0,   242,   158,   266,   242,   109,     0,     0,     0,     0,
     242,    11,     0,    39,     0,    41,     0,    38,    36,    37,
      46,     0,     0,     0,     0,     0,    71,     0,     0,     0,
      96,     0,     0,    82,   263,   119,   171,     0,     0,   244,
      94,     0,     0,   187,     0,     0,     0,     0,    75,     0,
       0,     0,     0,   179,   180,   181,   183,   178,     0,     0,
     212,   246,    88,   132,   134,   156,     0,   264,   111,   114,
      34,    12,    13,     0,     0,    33,    29,     0,    28,    35,
      50,    54,    60,    64,    67,    70,    73,    78,   100,    81,
       0,    98,   251,   224,   250,   170,     0,   186,     0,   192,
       0,   195,   196,   197,     0,     0,     0,     0,     0,   159,
      11,    15,     0,     0,    31,     0,   172,    97,   167,     0,
       0,     0,   135,   137,   136,     0,   215,     0,   216,   217,
     218,   219,   220,   221,   223,     0,   125,     0,     0,     0,
       0,   126,     0,     0,     0,   182,     0,   157,     0,    30,
      27,     0,   107,   107,     0,   162,   138,     0,     0,     0,
     222,   249,   185,   184,     0,   193,     0,   129,     0,   127,
     128,     0,    14,   168,     0,     0,   160,     0,     0,   107,
     107,     0,     0,   130,     0,   242,   242,     0,   163,   120,
       0,     0,     0,     0,     0,   188,     0,     0,   141,   145,
     161,   123,     0,   121,   122,   242,   242,   191,     0,     0,
       0,   124,   139,   143,   190,   142,   146,     0,     0,   140,
     144
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -329,     0,  -329,  -329,  -329,  -329,  -329,  -112,  -329,  -329,
    -329,  -329,  -329,  -329,  -329,  -329,  -329,    -7,  -329,  -329,
    -329,   325,  -169,   304,    99,   309,    95,   311,   101,   313,
     102,   317,   112,   189,  -329,  -329,   108,   -59,  -329,  -146,
    -329,   114,  -328,   -32,  -329,   143,   -17,  -329,   274,  -329,
     -87,  -329,   110,  -329,   259,  -329,   -24,  -329,   141,  -329,
     -50,  -329,  -329,  -329,  -329,  -329,  -329,  -329,  -329,  -329,
    -329,  -329,  -329,  -329,  -329,  -329,  -329,  -329,  -329,  -329,
    -329,  -329,  -329,  -329,  -329,  -329,  -329,  -329,  -329,  -329,
    -329,  -329,  -329,  -103,  -314,  -329,  -329,  -329,     8,  -269,
    -329,  -329,  -329,  -329,  -329,  -329,  -329,  -329,  -329,  -329,
     157,  -329,  -329,    16,  -329,  -329,   367,  -329,  -329,    31,
     -71,  -190,  -329,  -107,  -174,  -329,     9,   386,  -329,   -23,
     312,   -26,   -79,  -329
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,   159,   160,   161,   162,   163,   164,   165,   166,   167,
     168,   169,   170,   171,   172,   173,   316,   317,   258,   174,
     175,   176,   177,   261,   178,   262,   179,   263,   180,   264,
     181,   265,   182,    95,   183,   268,   184,   187,   289,   188,
     191,   192,   111,    75,   202,   127,   128,    40,    65,    66,
     112,   358,   193,    45,    72,    73,    46,   102,   142,   143,
     251,    12,   366,   217,    13,   239,    14,   240,   367,   368,
     457,   449,   369,   458,   450,    15,    67,    96,    54,    16,
     348,    17,   306,   370,   427,   371,   417,    18,    19,   391,
     278,   330,    20,    21,    22,   298,   219,   220,   221,   284,
     222,   223,   224,   225,   226,   227,   228,   229,   230,   231,
     232,   233,   373,   374,   375,    23,    24,    25,    26,   130,
     131,   113,   280,   120,   121,   334,   335,    33,   146,    59,
      60,    98,    99,   140
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      11,    27,    28,    29,    30,   132,    47,   260,   103,    61,
     248,   249,   234,   279,    53,    76,   218,   382,    68,   372,
      69,   139,   344,    11,   197,   206,   266,    10,    10,    10,
      10,    51,    79,    32,    10,    58,    62,    97,   114,    64,
      31,    58,    10,    58,    71,    56,   185,    56,   190,    10,
      57,   133,    57,    63,    51,    74,    58,    10,   243,   372,
     216,   141,    70,   106,   242,   303,   304,    56,    48,    49,
      50,   106,    35,   200,    51,   144,   126,    58,   386,   134,
     196,   147,   201,    86,    87,   119,   194,   259,   407,   203,
     430,   336,   320,   136,   106,   108,    51,   105,    36,   109,
      58,    58,   286,   238,    58,    38,   293,   294,   295,    58,
     106,   441,   107,    39,   108,    58,    41,    64,   109,   215,
      42,    43,    44,   327,   234,    71,   250,   110,   218,    58,
      84,    85,   349,   105,   296,   310,   297,    58,    55,   351,
     315,   105,   118,    74,   319,   119,   106,   282,   107,   285,
     108,   287,   288,   290,   109,   384,   136,   332,   108,   100,
     333,  -238,   109,   381,     1,    77,     2,     3,     4,     5,
     305,   105,   216,   307,   387,   337,   101,   276,   339,   313,
       6,     7,   104,   101,   106,   292,   107,     8,   108,     9,
      44,  -235,   109,    10,     1,   318,     2,     3,     4,     5,
      58,   429,    80,    81,   115,   281,   116,   283,    82,    83,
       6,     7,   136,   190,   108,   117,   137,     8,   109,     9,
     144,    88,    89,    10,   253,   106,   125,   254,   432,   255,
     122,   215,   256,   257,   136,   123,   108,    51,   124,   346,
     109,   394,   129,   428,   357,   129,    58,   309,   398,   399,
     135,   138,   309,   440,   331,   145,   309,   198,   383,   350,
     199,   236,   237,   241,   189,   244,   345,   245,   247,   246,
     403,   101,   252,    78,   269,   455,   456,    58,   270,   378,
     271,   379,   272,   459,   460,   359,   360,   361,   273,   274,
     275,   283,    58,   388,   318,   408,   390,   277,   119,     7,
     291,   299,   311,   301,   314,   362,   363,   364,   312,   105,
     338,    10,   340,   341,   435,   342,   343,   347,   352,   101,
     353,   406,   354,   355,   356,   376,   106,   377,   380,   447,
     385,   431,   397,   365,   396,   404,   401,   405,   409,   411,
     410,   454,   412,   442,   422,   421,   389,   283,   423,   437,
     443,   444,   424,   451,   438,   439,    52,    90,   322,   392,
     393,   321,    91,   436,    92,   323,    93,   324,   414,   415,
      94,   418,   267,   365,   452,   453,   326,   416,   448,   325,
     302,   328,   425,   426,   235,   402,   329,   308,   300,   400,
      37,   204,   395,    34,    51,   433,   434,    58,   419,   420,
     413,   445,   446,   205,   206,     0,   207,     0,   208,   209,
     210,   211,   212,     6,     0,   148,   149,   150,   213,   214,
       0,   195,     0,   151,     0,     0,    10,   152,   153,   154,
     155,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   156,     0,   157,     0,     0,   158,
       0,     0,    48,    49,    50,   148,   149,   150,     0,     0,
       0,     0,     0,   151,   119,     0,    10,   152,   153,   154,
     155,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   156,     0,   157,     0,     0,   158,
       0,     0,    48,    49,    50,   148,   149,   150,     0,     0,
     189,     0,     0,   151,     0,     0,    10,   152,   153,   154,
     155,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   156,   186,   157,     0,     0,   158,
       0,     0,    48,    49,    50,   148,   149,   150,     0,     0,
       0,     0,     0,   151,     0,     0,    10,   152,   153,   154,
     155,     0,     0,     0,     0,     0,     0,     0,    78,    79,
      80,    81,    82,    83,   156,     0,   157,     0,     0,   158,
       0,     0,    48,    49,    50,    84,    85,    86,    87,    88,
      89
};

static const yytype_int16 yycheck[] =
{
       0,     1,     2,     3,     4,    76,    30,   176,    58,    35,
     156,   157,   119,   203,    31,    47,   119,   345,    41,   333,
      43,   100,   291,    23,   111,    10,    37,    32,    32,    32,
      32,    31,    43,    30,    32,    35,    36,    54,    62,    39,
      50,    41,    32,    43,    44,    50,   105,    50,   107,    32,
      55,    77,    55,    51,    54,    50,    56,    32,   137,   373,
     119,    64,    64,    50,   135,   239,   240,    50,    58,    59,
      60,    50,    65,    68,    74,   101,    51,    77,   347,    96,
      67,   104,   114,    61,    62,    70,   109,   174,    67,   115,
     418,   281,   261,    52,    50,    54,    96,    37,     4,    58,
     100,   101,   209,   129,   104,     0,    39,    40,    41,   109,
      50,    67,    52,    50,    54,   115,    63,   117,    58,   119,
      67,    68,    63,   269,   231,   125,   158,    67,   231,   129,
      59,    60,   306,    37,    67,   247,    69,   137,    31,   313,
     252,    37,    67,    50,   256,    70,    50,   206,    52,   208,
      54,   210,   211,   212,    58,   345,    52,    67,    54,    50,
      70,     0,    58,    67,     3,    65,     5,     6,     7,     8,
     241,    37,   231,   244,   348,   282,    63,   200,   285,   250,
      19,    20,    57,    63,    50,    65,    52,    26,    54,    28,
      63,     0,    58,    32,     3,   254,     5,     6,     7,     8,
     200,    67,    44,    45,    65,   205,    51,   207,    46,    47,
      19,    20,    52,   272,    54,    56,    56,    26,    58,    28,
     246,    63,    64,    32,    49,    50,    56,    52,   418,    54,
      67,   231,    57,    58,    52,    67,    54,   237,    64,   298,
      58,    37,    38,   417,   331,    38,   246,   247,     6,     7,
      51,    51,   252,   427,   277,    72,   256,    67,   345,   309,
      67,    51,    56,    51,    66,    51,   292,    64,    57,    56,
     377,    63,    57,    42,    56,   449,   450,   277,    51,   338,
      56,   340,    56,   457,   458,     6,     7,     8,    53,    53,
      67,   291,   292,   352,   353,   382,   355,    63,    70,    20,
      12,    24,    51,    71,    54,    26,    27,    28,    53,    37,
      13,    32,    14,    67,   421,    67,    67,    12,    50,    63,
      56,   380,    53,    48,    67,    67,    50,    11,    13,   436,
      67,   418,    65,   333,    27,    48,    71,    67,    67,    13,
      67,   448,    51,   430,    48,   404,   353,   347,    67,    48,
      67,    67,   411,    67,   425,   426,    31,    53,   263,   359,
     360,   262,    53,   422,    53,   264,    53,   265,   392,   393,
      53,   397,   183,   373,   445,   446,   268,   394,   437,   267,
     237,   271,   414,   415,   125,   377,   272,   246,   231,   373,
      23,   117,   361,     7,   394,   419,   420,   397,   398,   399,
     391,   433,   434,     9,    10,    -1,    12,    -1,    14,    15,
      16,    17,    18,    19,    -1,    21,    22,    23,    24,    25,
      -1,   109,    -1,    29,    -1,    -1,    32,    33,    34,    35,
      36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    50,    -1,    52,    -1,    -1,    55,
      -1,    -1,    58,    59,    60,    21,    22,    23,    -1,    -1,
      -1,    -1,    -1,    29,    70,    -1,    32,    33,    34,    35,
      36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    50,    -1,    52,    -1,    -1,    55,
      -1,    -1,    58,    59,    60,    21,    22,    23,    -1,    -1,
      66,    -1,    -1,    29,    -1,    -1,    32,    33,    34,    35,
      36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    50,    51,    52,    -1,    -1,    55,
      -1,    -1,    58,    59,    60,    21,    22,    23,    -1,    -1,
      -1,    -1,    -1,    29,    -1,    -1,    32,    33,    34,    35,
      36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    42,    43,
      44,    45,    46,    47,    50,    -1,    52,    -1,    -1,    55,
      -1,    -1,    58,    59,    60,    59,    60,    61,    62,    63,
      64
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     5,     6,     7,     8,    19,    20,    26,    28,
      32,    74,   134,   137,   139,   148,   152,   154,   160,   161,
     165,   166,   167,   188,   189,   190,   191,    74,    74,    74,
      74,    50,    30,   200,   200,    65,     4,   189,     0,    50,
     120,    63,    67,    68,    63,   126,   129,   129,    58,    59,
      60,    74,    94,   119,   151,    31,    50,    55,    74,   202,
     203,   204,    74,    51,    74,   121,   122,   149,   202,   202,
      64,    74,   127,   128,    50,   116,   116,    65,    42,    43,
      44,    45,    46,    47,    59,    60,    61,    62,    63,    64,
      96,    98,   100,   102,   104,   106,   150,   119,   204,   205,
      50,    63,   130,   133,    57,    37,    50,    52,    54,    58,
      67,   115,   123,   194,   129,    65,    51,    56,    67,    70,
     196,   197,    67,    67,    64,    56,    51,   118,   119,    38,
     192,   193,   193,   204,   119,    51,    52,    56,    51,   205,
     206,    64,   131,   132,   204,    72,   201,   202,    21,    22,
      23,    29,    33,    34,    35,    36,    50,    52,    55,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    92,    93,    94,    95,    97,    99,
     101,   103,   105,   107,   109,   110,    51,   110,   112,    66,
     110,   113,   114,   125,   202,   203,    67,   123,    67,    67,
      68,   116,   117,   204,   121,     9,    10,    12,    14,    15,
      16,    17,    18,    24,    25,    74,   110,   136,   166,   169,
     170,   171,   173,   174,   175,   176,   177,   178,   179,   180,
     181,   182,   183,   184,   196,   127,    51,    56,   204,   138,
     140,    51,   193,   205,    51,    64,    56,    57,   112,   112,
     116,   133,    57,    49,    52,    54,    57,    58,    91,   123,
      95,    96,    98,   100,   102,   104,    37,   106,   108,    56,
      51,    56,    56,    53,    53,    67,   202,    63,   163,   194,
     195,    74,   110,    74,   172,   110,   196,   110,   110,   111,
     110,    12,    65,    39,    40,    41,    67,    69,   168,    24,
     183,    71,   118,   197,   197,   193,   155,   193,   131,    74,
      80,    51,    53,   193,    54,    80,    89,    90,   110,    80,
      95,    97,    99,   101,   103,   105,   109,   112,   125,   114,
     164,   202,    67,    70,   198,   199,   194,   196,    13,   196,
      14,    67,    67,    67,   172,   204,   110,    12,   153,   197,
     133,   197,    50,    56,    53,    48,    67,   123,   124,     6,
       7,     8,    26,    27,    28,    74,   135,   141,   142,   145,
     156,   158,   167,   185,   186,   187,    67,    11,   110,   110,
      13,    67,   115,   123,   194,    67,   172,   197,   110,    90,
     110,   162,    74,    74,    37,   192,    27,    65,     6,     7,
     186,    71,   171,   196,    48,    67,   110,    67,   123,    67,
      67,    13,    51,   199,   129,   129,   119,   159,   204,    74,
      74,   110,    48,    67,   110,   116,   116,   157,   197,    67,
     115,   123,   194,   129,   129,   196,   110,    48,   193,   193,
     197,    67,   123,    67,    67,   116,   116,   196,   110,   144,
     147,    67,   193,   193,   196,   197,   197,   143,   146,   197,
     197
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    73,    74,    75,    76,    77,    78,    79,    79,    79,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      88,    88,    88,    88,    88,    88,    88,    89,    89,    90,
      90,    91,    92,    92,    92,    92,    92,    92,    92,    92,
      93,    93,    94,    94,    94,    95,    95,    96,    96,    97,
      97,    98,    98,    99,    99,   100,   100,   100,   100,   101,
     101,   102,   102,   103,   103,   104,   105,   105,   106,   107,
     107,   108,   109,   109,   110,   111,   111,   112,   112,   113,
     114,   114,   115,   116,   116,   117,   117,   118,   118,   119,
     120,   120,   121,   121,   122,   123,   123,   124,   124,   125,
     125,   126,   126,   127,   127,   128,   129,   129,   130,   130,
     131,   131,   132,   133,   133,   134,   134,   134,   134,   134,
     135,   135,   135,   135,   135,   136,   136,   136,   136,   136,
     136,   138,   137,   140,   139,   141,   141,   141,   141,   143,
     142,   144,   142,   146,   145,   147,   145,   149,   148,   150,
     150,   150,   150,   150,   150,   151,   153,   152,   155,   154,
     157,   156,   159,   158,   160,   160,   160,   162,   161,   163,
     161,   164,   161,   165,   165,   165,   166,   167,   168,   168,
     168,   168,   169,   170,   171,   171,   171,   172,   173,   174,
     175,   175,   176,   177,   178,   179,   180,   181,   182,   182,
     182,   182,   182,   182,   182,   182,   182,   182,   182,   182,
     182,   183,   183,   184,   184,   185,   185,   185,   185,   185,
     185,   186,   186,   187,   187,   188,   188,   188,   188,   188,
     188,   188,   188,   188,   188,   189,   189,   190,   190,   191,
     192,   193,   193,   194,   195,   195,   196,   197,   197,   198,
     199,   199,   200,   201,   201,   202,   203,   203,   204,   204,
     204,   204,   204,   204,   204,   205,   205,   206,   206
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_int8 yyr2[] =
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
       6,     0,     7,     0,     7,     1,     1,     1,     2,     0,
       8,     0,     7,     0,     8,     0,     7,     0,     5,     1,
       1,     1,     1,     1,     1,     1,     0,     9,     0,     8,
       0,     5,     0,     4,     1,     1,     0,     0,    10,     0,
       7,     0,     8,     5,     5,     3,     2,     2,     1,     1,
       1,     1,     4,     2,     5,     5,     3,     1,     7,     1,
       9,     8,     3,     5,     1,     3,     3,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     2,     1,     0,     1,     1,     1,     1,     1,
       1,     1,     2,     1,     0,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     2,     1,     0,     1,
       2,     1,     0,     2,     1,     0,     3,     1,     1,     3,
       1,     1,     2,     1,     0,     3,     1,     3,     1,     2,
       2,     3,     3,     4,     5,     1,     3,     1,     0
};


/* YYDPREC[RULE-NUM] -- Dynamic precedence of rule #RULE-NUM (0 if none).  */
static const yytype_int8 yydprec[] =
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
       0,     0,     0,     0,     0,     0,     0,     0,     0
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
       0,     0,     0,     0,     0,     0,     0,     0,     0
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
       0,     0,     0,     0,     0,     0,     0,     0,     0
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
       0,     0,     0,     0,     0,     0,     0,     5,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     1,     0,     0,     0,
       0,     0,     0,     9,     0,     0,     0,     0,     0,     0,
       3,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    11,     0,    13,     0,     0,     0,
      15,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     7,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    17,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    19,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0
};

/* YYCONFL[I] -- lists of conflicting rule numbers, each terminated by
   0, pointed into by YYCONFLP.  */
static const short yyconfl[] =
{
       0,   114,     0,   107,     0,   260,     0,   114,     0,   114,
       0,   240,     0,   240,     0,   240,     0,    11,     0,   114,
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
#line 161 "bi/parser.ypp"
            { ((*yyvalp).valName) = new bi::Name((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valString)); }
#line 1606 "bi/parser.cpp"
    break;

  case 3:
#line 170 "bi/parser.ypp"
                    { ((*yyvalp).valExpression) = new bi::Literal<bool>((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valString), make_loc((*yylocp))); }
#line 1612 "bi/parser.cpp"
    break;

  case 4:
#line 174 "bi/parser.ypp"
                   { ((*yyvalp).valExpression) = new bi::Literal<int64_t>((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valString), make_loc((*yylocp))); }
#line 1618 "bi/parser.cpp"
    break;

  case 5:
#line 178 "bi/parser.ypp"
                    { ((*yyvalp).valExpression) = new bi::Literal<double>((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valString), make_loc((*yylocp))); }
#line 1624 "bi/parser.cpp"
    break;

  case 6:
#line 182 "bi/parser.ypp"
                      { ((*yyvalp).valExpression) = new bi::Literal<const char*>((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valString), make_loc((*yylocp))); }
#line 1630 "bi/parser.cpp"
    break;

  case 11:
#line 193 "bi/parser.ypp"
                                       { ((*yyvalp).valExpression) = new bi::NamedExpression((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 1636 "bi/parser.cpp"
    break;

  case 12:
#line 197 "bi/parser.ypp"
                               { ((*yyvalp).valExpression) = new bi::Parentheses((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1642 "bi/parser.cpp"
    break;

  case 13:
#line 201 "bi/parser.ypp"
                               { ((*yyvalp).valExpression) = new bi::Sequence((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1648 "bi/parser.cpp"
    break;

  case 14:
#line 205 "bi/parser.ypp"
                                                              { ((*yyvalp).valExpression) = new bi::Cast(new bi::NamedType(false, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1654 "bi/parser.cpp"
    break;

  case 15:
#line 209 "bi/parser.ypp"
                                                           { ((*yyvalp).valExpression) = new bi::LambdaFunction((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 1660 "bi/parser.cpp"
    break;

  case 16:
#line 213 "bi/parser.ypp"
            { ((*yyvalp).valExpression) = new bi::This(make_loc((*yylocp))); }
#line 1666 "bi/parser.cpp"
    break;

  case 17:
#line 217 "bi/parser.ypp"
             { ((*yyvalp).valExpression) = new bi::Super(make_loc((*yylocp))); }
#line 1672 "bi/parser.cpp"
    break;

  case 18:
#line 221 "bi/parser.ypp"
           { ((*yyvalp).valExpression) = new bi::Nil(make_loc((*yylocp))); }
#line 1678 "bi/parser.cpp"
    break;

  case 27:
#line 236 "bi/parser.ypp"
                                      { ((*yyvalp).valExpression) = new bi::Range((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1684 "bi/parser.cpp"
    break;

  case 28:
#line 237 "bi/parser.ypp"
                                      { ((*yyvalp).valExpression) = new bi::Index((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1690 "bi/parser.cpp"
    break;

  case 30:
#line 242 "bi/parser.ypp"
                                       { ((*yyvalp).valExpression) = new bi::ExpressionList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1696 "bi/parser.cpp"
    break;

  case 31:
#line 246 "bi/parser.ypp"
                          { ((*yyvalp).valExpression) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1702 "bi/parser.cpp"
    break;

  case 33:
#line 251 "bi/parser.ypp"
                                         { ((*yyvalp).valExpression) = new bi::Member((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1708 "bi/parser.cpp"
    break;

  case 34:
#line 252 "bi/parser.ypp"
                                         { ((*yyvalp).valExpression) = new bi::Global((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1714 "bi/parser.cpp"
    break;

  case 35:
#line 253 "bi/parser.ypp"
                                         { ((*yyvalp).valExpression) = new bi::Member((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1720 "bi/parser.cpp"
    break;

  case 36:
#line 254 "bi/parser.ypp"
                                         { ((*yyvalp).valExpression) = new bi::Slice((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1726 "bi/parser.cpp"
    break;

  case 37:
#line 255 "bi/parser.ypp"
                                         { ((*yyvalp).valExpression) = new bi::Call((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1732 "bi/parser.cpp"
    break;

  case 38:
#line 256 "bi/parser.ypp"
                                         { ((*yyvalp).valExpression) = new bi::Get((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1738 "bi/parser.cpp"
    break;

  case 39:
#line 257 "bi/parser.ypp"
                                         { ((*yyvalp).valExpression) = new bi::Spin((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1744 "bi/parser.cpp"
    break;

  case 41:
#line 262 "bi/parser.ypp"
                              { ((*yyvalp).valExpression) = new bi::Query((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1750 "bi/parser.cpp"
    break;

  case 42:
#line 266 "bi/parser.ypp"
           { ((*yyvalp).valName) = new bi::Name("+"); }
#line 1756 "bi/parser.cpp"
    break;

  case 43:
#line 267 "bi/parser.ypp"
           { ((*yyvalp).valName) = new bi::Name("-"); }
#line 1762 "bi/parser.cpp"
    break;

  case 44:
#line 268 "bi/parser.ypp"
           { ((*yyvalp).valName) = new bi::Name("!"); }
#line 1768 "bi/parser.cpp"
    break;

  case 46:
#line 273 "bi/parser.ypp"
                                         { ((*yyvalp).valExpression) = new bi::UnaryCall((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1774 "bi/parser.cpp"
    break;

  case 47:
#line 277 "bi/parser.ypp"
           { ((*yyvalp).valName) = new bi::Name("*"); }
#line 1780 "bi/parser.cpp"
    break;

  case 48:
#line 278 "bi/parser.ypp"
           { ((*yyvalp).valName) = new bi::Name("/"); }
#line 1786 "bi/parser.cpp"
    break;

  case 50:
#line 283 "bi/parser.ypp"
                                                                           { ((*yyvalp).valExpression) = new bi::BinaryCall((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1792 "bi/parser.cpp"
    break;

  case 51:
#line 287 "bi/parser.ypp"
           { ((*yyvalp).valName) = new bi::Name("+"); }
#line 1798 "bi/parser.cpp"
    break;

  case 52:
#line 288 "bi/parser.ypp"
           { ((*yyvalp).valName) = new bi::Name("-"); }
#line 1804 "bi/parser.cpp"
    break;

  case 54:
#line 293 "bi/parser.ypp"
                                                                       { ((*yyvalp).valExpression) = new bi::BinaryCall((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1810 "bi/parser.cpp"
    break;

  case 55:
#line 297 "bi/parser.ypp"
             { ((*yyvalp).valName) = new bi::Name("<"); }
#line 1816 "bi/parser.cpp"
    break;

  case 56:
#line 298 "bi/parser.ypp"
             { ((*yyvalp).valName) = new bi::Name(">"); }
#line 1822 "bi/parser.cpp"
    break;

  case 57:
#line 299 "bi/parser.ypp"
             { ((*yyvalp).valName) = new bi::Name("<="); }
#line 1828 "bi/parser.cpp"
    break;

  case 58:
#line 300 "bi/parser.ypp"
             { ((*yyvalp).valName) = new bi::Name(">="); }
#line 1834 "bi/parser.cpp"
    break;

  case 60:
#line 310 "bi/parser.ypp"
                                                                               { ((*yyvalp).valExpression) = new bi::BinaryCall((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1840 "bi/parser.cpp"
    break;

  case 61:
#line 314 "bi/parser.ypp"
             { ((*yyvalp).valName) = new bi::Name("=="); }
#line 1846 "bi/parser.cpp"
    break;

  case 62:
#line 315 "bi/parser.ypp"
             { ((*yyvalp).valName) = new bi::Name("!="); }
#line 1852 "bi/parser.cpp"
    break;

  case 64:
#line 320 "bi/parser.ypp"
                                                                   { ((*yyvalp).valExpression) = new bi::BinaryCall((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1858 "bi/parser.cpp"
    break;

  case 65:
#line 324 "bi/parser.ypp"
              { ((*yyvalp).valName) = new bi::Name("&&"); }
#line 1864 "bi/parser.cpp"
    break;

  case 67:
#line 329 "bi/parser.ypp"
                                                                       { ((*yyvalp).valExpression) = new bi::BinaryCall((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1870 "bi/parser.cpp"
    break;

  case 68:
#line 333 "bi/parser.ypp"
             { ((*yyvalp).valName) = new bi::Name("||"); }
#line 1876 "bi/parser.cpp"
    break;

  case 70:
#line 338 "bi/parser.ypp"
                                                                        { ((*yyvalp).valExpression) = new bi::BinaryCall((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1882 "bi/parser.cpp"
    break;

  case 71:
#line 342 "bi/parser.ypp"
                 { ((*yyvalp).valName) = new bi::Name("<-"); }
#line 1888 "bi/parser.cpp"
    break;

  case 73:
#line 347 "bi/parser.ypp"
                                                               { ((*yyvalp).valExpression) = new bi::Assign((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1894 "bi/parser.cpp"
    break;

  case 76:
#line 356 "bi/parser.ypp"
                  { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1900 "bi/parser.cpp"
    break;

  case 78:
#line 361 "bi/parser.ypp"
                                      { ((*yyvalp).valExpression) = new bi::ExpressionList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1906 "bi/parser.cpp"
    break;

  case 79:
#line 365 "bi/parser.ypp"
                   { ((*yyvalp).valExpression) = new bi::Span((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1912 "bi/parser.cpp"
    break;

  case 81:
#line 370 "bi/parser.ypp"
                                     { ((*yyvalp).valExpression) = new bi::ExpressionList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1918 "bi/parser.cpp"
    break;

  case 82:
#line 374 "bi/parser.ypp"
                         { ((*yyvalp).valExpression) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1924 "bi/parser.cpp"
    break;

  case 83:
#line 378 "bi/parser.ypp"
                              { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1930 "bi/parser.cpp"
    break;

  case 84:
#line 379 "bi/parser.ypp"
                              { ((*yyvalp).valExpression) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1936 "bi/parser.cpp"
    break;

  case 86:
#line 384 "bi/parser.ypp"
                  { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1942 "bi/parser.cpp"
    break;

  case 88:
#line 389 "bi/parser.ypp"
                                    { ((*yyvalp).valExpression) = new bi::ExpressionList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1948 "bi/parser.cpp"
    break;

  case 89:
#line 393 "bi/parser.ypp"
                     { ((*yyvalp).valExpression) = new bi::Parameter(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), make_loc((*yylocp))); }
#line 1954 "bi/parser.cpp"
    break;

  case 90:
#line 397 "bi/parser.ypp"
                           { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1960 "bi/parser.cpp"
    break;

  case 91:
#line 398 "bi/parser.ypp"
                           { ((*yyvalp).valExpression) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1966 "bi/parser.cpp"
    break;

  case 93:
#line 403 "bi/parser.ypp"
                              { ((*yyvalp).valExpression) = new bi::ExpressionList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1972 "bi/parser.cpp"
    break;

  case 94:
#line 407 "bi/parser.ypp"
                                    { ((*yyvalp).valExpression) = new bi::Parameter(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 1978 "bi/parser.cpp"
    break;

  case 95:
#line 411 "bi/parser.ypp"
                               { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1984 "bi/parser.cpp"
    break;

  case 96:
#line 412 "bi/parser.ypp"
                               { ((*yyvalp).valExpression) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 1990 "bi/parser.cpp"
    break;

  case 98:
#line 417 "bi/parser.ypp"
                 { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 1996 "bi/parser.cpp"
    break;

  case 99:
#line 421 "bi/parser.ypp"
                     { ((*yyvalp).valInt) = 1; }
#line 2002 "bi/parser.cpp"
    break;

  case 100:
#line 422 "bi/parser.ypp"
                     { ((*yyvalp).valInt) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valInt) + 1; }
#line 2008 "bi/parser.cpp"
    break;

  case 101:
#line 426 "bi/parser.ypp"
                            { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 2014 "bi/parser.cpp"
    break;

  case 102:
#line 427 "bi/parser.ypp"
                            { ((*yyvalp).valExpression) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression); }
#line 2020 "bi/parser.cpp"
    break;

  case 104:
#line 432 "bi/parser.ypp"
                                { ((*yyvalp).valExpression) = new bi::ExpressionList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2026 "bi/parser.cpp"
    break;

  case 105:
#line 436 "bi/parser.ypp"
            { ((*yyvalp).valExpression) = new bi::Generic(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valName), empty_type((*yylocp)), make_loc((*yylocp))); }
#line 2032 "bi/parser.cpp"
    break;

  case 107:
#line 441 "bi/parser.ypp"
                { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 2038 "bi/parser.cpp"
    break;

  case 108:
#line 445 "bi/parser.ypp"
                                     { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 2044 "bi/parser.cpp"
    break;

  case 109:
#line 446 "bi/parser.ypp"
                                     { ((*yyvalp).valType) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType); }
#line 2050 "bi/parser.cpp"
    break;

  case 111:
#line 451 "bi/parser.ypp"
                                                  { ((*yyvalp).valType) = new bi::TypeList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2056 "bi/parser.cpp"
    break;

  case 114:
#line 460 "bi/parser.ypp"
                         { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 2062 "bi/parser.cpp"
    break;

  case 115:
#line 469 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2068 "bi/parser.cpp"
    break;

  case 116:
#line 470 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2074 "bi/parser.cpp"
    break;

  case 117:
#line 471 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 2080 "bi/parser.cpp"
    break;

  case 118:
#line 472 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), new bi::ArrayType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression)->width(), make_loc((*yylocp))), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2086 "bi/parser.cpp"
    break;

  case 119:
#line 473 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::GlobalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), new bi::ArrayType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression)->width(), make_loc((*yylocp))), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2092 "bi/parser.cpp"
    break;

  case 120:
#line 477 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2098 "bi/parser.cpp"
    break;

  case 121:
#line 478 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2104 "bi/parser.cpp"
    break;

  case 122:
#line 479 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 2110 "bi/parser.cpp"
    break;

  case 123:
#line 480 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), new bi::ArrayType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression)->width(), make_loc((*yylocp))), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2116 "bi/parser.cpp"
    break;

  case 124:
#line 481 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::MemberVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), new bi::ArrayType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression)->width(), make_loc((*yylocp))), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2122 "bi/parser.cpp"
    break;

  case 125:
#line 485 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::LocalVariable(bi::AUTO, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), empty_type((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 2128 "bi/parser.cpp"
    break;

  case 126:
#line 486 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::LocalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2134 "bi/parser.cpp"
    break;

  case 127:
#line 487 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::LocalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2140 "bi/parser.cpp"
    break;

  case 128:
#line 488 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::LocalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), empty_expr((*yylocp)), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_doc_loc((*yylocp))); }
#line 2146 "bi/parser.cpp"
    break;

  case 129:
#line 489 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::LocalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), new bi::ArrayType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression)->width(), make_loc((*yylocp))), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2152 "bi/parser.cpp"
    break;

  case 130:
#line 490 "bi/parser.ypp"
                                            { push_raw(); ((*yyvalp).valStatement) = new bi::LocalVariable(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), new bi::ArrayType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression)->width(), make_loc((*yylocp))), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), make_doc_loc((*yylocp))); }
#line 2158 "bi/parser.cpp"
    break;

  case 131:
#line 494 "bi/parser.ypp"
                                                                      { push_raw(); }
#line 2164 "bi/parser.cpp"
    break;

  case 132:
#line 494 "bi/parser.ypp"
                                                                                                       { ((*yyvalp).valStatement) = new bi::Function(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2170 "bi/parser.cpp"
    break;

  case 133:
#line 498 "bi/parser.ypp"
                                                                   { push_raw(); }
#line 2176 "bi/parser.cpp"
    break;

  case 134:
#line 498 "bi/parser.ypp"
                                                                                                    { ((*yyvalp).valStatement) = new bi::Fiber(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2182 "bi/parser.cpp"
    break;

  case 135:
#line 502 "bi/parser.ypp"
                      { ((*yyvalp).valAnnotation) = bi::ABSTRACT; }
#line 2188 "bi/parser.cpp"
    break;

  case 136:
#line 503 "bi/parser.ypp"
                      { ((*yyvalp).valAnnotation) = bi::FINAL; }
#line 2194 "bi/parser.cpp"
    break;

  case 137:
#line 504 "bi/parser.ypp"
                      { ((*yyvalp).valAnnotation) = bi::OVERRIDE; }
#line 2200 "bi/parser.cpp"
    break;

  case 138:
#line 505 "bi/parser.ypp"
                      { ((*yyvalp).valAnnotation) = bi::Annotation(bi::FINAL|bi::OVERRIDE); }
#line 2206 "bi/parser.cpp"
    break;

  case 139:
#line 511 "bi/parser.ypp"
                                                                                                 { push_raw(); }
#line 2212 "bi/parser.cpp"
    break;

  case 140:
#line 511 "bi/parser.ypp"
                                                                                                                                  { ((*yyvalp).valStatement) = new bi::MemberFunction((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-7)].yystate.yysemantics.yysval.valAnnotation), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2218 "bi/parser.cpp"
    break;

  case 141:
#line 512 "bi/parser.ypp"
                                                                      { push_raw(); }
#line 2224 "bi/parser.cpp"
    break;

  case 142:
#line 512 "bi/parser.ypp"
                                                                                                                                  { ((*yyvalp).valStatement) = new bi::MemberFunction(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2230 "bi/parser.cpp"
    break;

  case 143:
#line 516 "bi/parser.ypp"
                                                                                              { push_raw(); }
#line 2236 "bi/parser.cpp"
    break;

  case 144:
#line 516 "bi/parser.ypp"
                                                                                                                               { ((*yyvalp).valStatement) = new bi::MemberFiber((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-7)].yystate.yysemantics.yysval.valAnnotation), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2242 "bi/parser.cpp"
    break;

  case 145:
#line 517 "bi/parser.ypp"
                                                                   { push_raw(); }
#line 2248 "bi/parser.cpp"
    break;

  case 146:
#line 517 "bi/parser.ypp"
                                                                                                                               { ((*yyvalp).valStatement) = new bi::MemberFiber(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2254 "bi/parser.cpp"
    break;

  case 147:
#line 521 "bi/parser.ypp"
                           { push_raw(); }
#line 2260 "bi/parser.cpp"
    break;

  case 148:
#line 521 "bi/parser.ypp"
                                                            { ((*yyvalp).valStatement) = new bi::Program((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2266 "bi/parser.cpp"
    break;

  case 156:
#line 538 "bi/parser.ypp"
                                                                                { push_raw(); }
#line 2272 "bi/parser.cpp"
    break;

  case 157:
#line 538 "bi/parser.ypp"
                                                                                                                 { ((*yyvalp).valStatement) = new bi::BinaryOperator(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2278 "bi/parser.cpp"
    break;

  case 158:
#line 542 "bi/parser.ypp"
                                                                     { push_raw(); }
#line 2284 "bi/parser.cpp"
    break;

  case 159:
#line 542 "bi/parser.ypp"
                                                                                                      { ((*yyvalp).valStatement) = new bi::UnaryOperator(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2290 "bi/parser.cpp"
    break;

  case 160:
#line 546 "bi/parser.ypp"
                                 { push_raw(); }
#line 2296 "bi/parser.cpp"
    break;

  case 161:
#line 546 "bi/parser.ypp"
                                                                  { ((*yyvalp).valStatement) = new bi::AssignmentOperator((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2302 "bi/parser.cpp"
    break;

  case 162:
#line 550 "bi/parser.ypp"
                           { push_raw(); }
#line 2308 "bi/parser.cpp"
    break;

  case 163:
#line 550 "bi/parser.ypp"
                                                            { ((*yyvalp).valStatement) = new bi::ConversionOperator((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2314 "bi/parser.cpp"
    break;

  case 164:
#line 554 "bi/parser.ypp"
                { ((*yyvalp).valAnnotation) = bi::FINAL; }
#line 2320 "bi/parser.cpp"
    break;

  case 165:
#line 555 "bi/parser.ypp"
                { ((*yyvalp).valAnnotation) = bi::ABSTRACT; }
#line 2326 "bi/parser.cpp"
    break;

  case 166:
#line 556 "bi/parser.ypp"
                { ((*yyvalp).valAnnotation) = bi::NONE; }
#line 2332 "bi/parser.cpp"
    break;

  case 167:
#line 560 "bi/parser.ypp"
                                                                                                          { push_raw(); }
#line 2338 "bi/parser.cpp"
    break;

  case 168:
#line 560 "bi/parser.ypp"
                                                                                                                                                 { ((*yyvalp).valStatement) = new bi::Class((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-9)].yystate.yysemantics.yysval.valAnnotation), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-7)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), false, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2344 "bi/parser.cpp"
    break;

  case 169:
#line 561 "bi/parser.ypp"
                                                                        { push_raw(); }
#line 2350 "bi/parser.cpp"
    break;

  case 170:
#line 561 "bi/parser.ypp"
                                                                                                                                                 { ((*yyvalp).valStatement) = new bi::Class((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-6)].yystate.yysemantics.yysval.valAnnotation), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valExpression), empty_type((*yylocp)), false, empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_doc_loc((*yylocp))); }
#line 2356 "bi/parser.cpp"
    break;

  case 171:
#line 562 "bi/parser.ypp"
                                                                   { push_raw(); }
#line 2362 "bi/parser.cpp"
    break;

  case 172:
#line 562 "bi/parser.ypp"
                                                                                                                                                 { ((*yyvalp).valStatement) = new bi::Class((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-7)].yystate.yysemantics.yysval.valAnnotation), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-4)].yystate.yysemantics.yysval.valExpression), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), true, empty_expr((*yylocp)), empty_stmt((*yylocp)), make_doc_loc((*yylocp))); }
#line 2368 "bi/parser.cpp"
    break;

  case 173:
#line 566 "bi/parser.ypp"
                                    { push_raw(); ((*yyvalp).valStatement) = new bi::Basic(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), false, make_doc_loc((*yylocp))); }
#line 2374 "bi/parser.cpp"
    break;

  case 174:
#line 567 "bi/parser.ypp"
                                    { push_raw(); ((*yyvalp).valStatement) = new bi::Basic(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valName), empty_expr((*yylocp)), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), true, make_doc_loc((*yylocp))); }
#line 2380 "bi/parser.cpp"
    break;

  case 175:
#line 568 "bi/parser.ypp"
                                    { push_raw(); ((*yyvalp).valStatement) = new bi::Basic(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valName), empty_expr((*yylocp)), empty_type((*yylocp)), false, make_doc_loc((*yylocp))); }
#line 2386 "bi/parser.cpp"
    break;

  case 176:
#line 572 "bi/parser.ypp"
                         { push_raw(); ((*yyvalp).valStatement) = new bi::Raw(new bi::Name("cpp"), pop_raw(), make_loc((*yylocp))); }
#line 2392 "bi/parser.cpp"
    break;

  case 177:
#line 576 "bi/parser.ypp"
                         { push_raw(); ((*yyvalp).valStatement) = new bi::Raw(new bi::Name("hpp"), pop_raw(), make_loc((*yylocp))); }
#line 2398 "bi/parser.cpp"
    break;

  case 178:
#line 580 "bi/parser.ypp"
                      { ((*yyvalp).valName) = new bi::Name("~"); }
#line 2404 "bi/parser.cpp"
    break;

  case 179:
#line 581 "bi/parser.ypp"
                      { ((*yyvalp).valName) = new bi::Name("<~"); }
#line 2410 "bi/parser.cpp"
    break;

  case 180:
#line 582 "bi/parser.ypp"
                      { ((*yyvalp).valName) = new bi::Name("~>"); }
#line 2416 "bi/parser.cpp"
    break;

  case 181:
#line 583 "bi/parser.ypp"
                      { ((*yyvalp).valName) = new bi::Name("<-?"); }
#line 2422 "bi/parser.cpp"
    break;

  case 182:
#line 587 "bi/parser.ypp"
                                                 { ((*yyvalp).valStatement) = new bi::Assume((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2428 "bi/parser.cpp"
    break;

  case 183:
#line 591 "bi/parser.ypp"
                      { ((*yyvalp).valStatement) = new bi::ExpressionStatement((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2434 "bi/parser.cpp"
    break;

  case 184:
#line 595 "bi/parser.ypp"
                                        { ((*yyvalp).valStatement) = new bi::If((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valStatement), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2440 "bi/parser.cpp"
    break;

  case 185:
#line 596 "bi/parser.ypp"
                                        { ((*yyvalp).valStatement) = new bi::If((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valStatement), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2446 "bi/parser.cpp"
    break;

  case 186:
#line 597 "bi/parser.ypp"
                                        { ((*yyvalp).valStatement) = new bi::If((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), empty_stmt((*yylocp)), make_loc((*yylocp))); }
#line 2452 "bi/parser.cpp"
    break;

  case 187:
#line 601 "bi/parser.ypp"
                          { ((*yyvalp).valStatement) = new bi::LocalVariable((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valName), new bi::NamedType(new bi::Name("Integer")), make_loc((*yylocp))); }
#line 2458 "bi/parser.cpp"
    break;

  case 188:
#line 605 "bi/parser.ypp"
                                                                             { ((*yyvalp).valStatement) = new bi::For(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valStatement), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2464 "bi/parser.cpp"
    break;

  case 189:
#line 609 "bi/parser.ypp"
               { ((*yyvalp).valAnnotation) = bi::DYNAMIC; }
#line 2470 "bi/parser.cpp"
    break;

  case 190:
#line 615 "bi/parser.ypp"
                                                                                                          { ((*yyvalp).valStatement) = new bi::Parallel((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-8)].yystate.yysemantics.yysval.valAnnotation), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valStatement), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2476 "bi/parser.cpp"
    break;

  case 191:
#line 616 "bi/parser.ypp"
                                                                                                          { ((*yyvalp).valStatement) = new bi::Parallel(bi::NONE, (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-5)].yystate.yysemantics.yysval.valStatement), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2482 "bi/parser.cpp"
    break;

  case 192:
#line 620 "bi/parser.ypp"
                               { ((*yyvalp).valStatement) = new bi::While((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2488 "bi/parser.cpp"
    break;

  case 193:
#line 624 "bi/parser.ypp"
                                      { ((*yyvalp).valStatement) = new bi::DoWhile((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valStatement), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2494 "bi/parser.cpp"
    break;

  case 194:
#line 628 "bi/parser.ypp"
              { ((*yyvalp).valStatement) = new bi::Block((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2500 "bi/parser.cpp"
    break;

  case 195:
#line 632 "bi/parser.ypp"
                             { ((*yyvalp).valStatement) = new bi::Assert((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2506 "bi/parser.cpp"
    break;

  case 196:
#line 636 "bi/parser.ypp"
                                      { ((*yyvalp).valStatement) = new bi::Return((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2512 "bi/parser.cpp"
    break;

  case 197:
#line 640 "bi/parser.ypp"
                            { ((*yyvalp).valStatement) = new bi::Yield((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valExpression), make_loc((*yylocp))); }
#line 2518 "bi/parser.cpp"
    break;

  case 212:
#line 661 "bi/parser.ypp"
                            { ((*yyvalp).valStatement) = new bi::StatementList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2524 "bi/parser.cpp"
    break;

  case 214:
#line 666 "bi/parser.ypp"
                  { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2530 "bi/parser.cpp"
    break;

  case 222:
#line 680 "bi/parser.ypp"
                                        { ((*yyvalp).valStatement) = new bi::StatementList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2536 "bi/parser.cpp"
    break;

  case 224:
#line 685 "bi/parser.ypp"
                        { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2542 "bi/parser.cpp"
    break;

  case 236:
#line 703 "bi/parser.ypp"
                                      { ((*yyvalp).valStatement) = new bi::StatementList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2548 "bi/parser.cpp"
    break;

  case 238:
#line 708 "bi/parser.ypp"
                       { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2554 "bi/parser.cpp"
    break;

  case 239:
#line 712 "bi/parser.ypp"
                                { compiler->setRoot((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valStatement)); }
#line 2560 "bi/parser.cpp"
    break;

  case 240:
#line 716 "bi/parser.ypp"
                     { ((*yyvalp).valType) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType); }
#line 2566 "bi/parser.cpp"
    break;

  case 242:
#line 721 "bi/parser.ypp"
                   { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 2572 "bi/parser.cpp"
    break;

  case 243:
#line 725 "bi/parser.ypp"
                          { ((*yyvalp).valExpression) = (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valExpression); }
#line 2578 "bi/parser.cpp"
    break;

  case 245:
#line 730 "bi/parser.ypp"
             { ((*yyvalp).valExpression) = empty_expr((*yylocp)); }
#line 2584 "bi/parser.cpp"
    break;

  case 246:
#line 734 "bi/parser.ypp"
                                   { ((*yyvalp).valStatement) = new bi::Braces((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2590 "bi/parser.cpp"
    break;

  case 248:
#line 739 "bi/parser.ypp"
              { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2596 "bi/parser.cpp"
    break;

  case 249:
#line 743 "bi/parser.ypp"
                                         { ((*yyvalp).valStatement) = new bi::Braces((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valStatement), make_loc((*yylocp))); }
#line 2602 "bi/parser.cpp"
    break;

  case 251:
#line 748 "bi/parser.ypp"
                    { ((*yyvalp).valStatement) = empty_stmt((*yylocp)); }
#line 2608 "bi/parser.cpp"
    break;

  case 253:
#line 761 "bi/parser.ypp"
           { ((*yyvalp).valBool) = true; }
#line 2614 "bi/parser.cpp"
    break;

  case 254:
#line 762 "bi/parser.ypp"
           { ((*yyvalp).valBool) = false; }
#line 2620 "bi/parser.cpp"
    break;

  case 255:
#line 766 "bi/parser.ypp"
                                                     { ((*yyvalp).valType) = new bi::NamedType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valBool), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valName), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2626 "bi/parser.cpp"
    break;

  case 257:
#line 771 "bi/parser.ypp"
                         { ((*yyvalp).valType) = new bi::TupleType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2632 "bi/parser.cpp"
    break;

  case 259:
#line 776 "bi/parser.ypp"
                                                           { ((*yyvalp).valType) = new bi::OptionalType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2638 "bi/parser.cpp"
    break;

  case 260:
#line 777 "bi/parser.ypp"
                                                           { ((*yyvalp).valType) = new bi::FiberType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valType), empty_type((*yylocp)), make_loc((*yylocp))); }
#line 2644 "bi/parser.cpp"
    break;

  case 261:
#line 778 "bi/parser.ypp"
                                                           { ((*yyvalp).valType) = new bi::FiberType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2650 "bi/parser.cpp"
    break;

  case 262:
#line 779 "bi/parser.ypp"
                                                           { ((*yyvalp).valType) = new bi::MemberType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2656 "bi/parser.cpp"
    break;

  case 263:
#line 780 "bi/parser.ypp"
                                                           { ((*yyvalp).valType) = new bi::ArrayType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-3)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-1)].yystate.yysemantics.yysval.valInt), make_loc((*yylocp))); }
#line 2662 "bi/parser.cpp"
    break;

  case 264:
#line 781 "bi/parser.ypp"
                                                           { ((*yyvalp).valType) = new bi::FunctionType((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2668 "bi/parser.cpp"
    break;

  case 266:
#line 786 "bi/parser.ypp"
                          { ((*yyvalp).valType) = new bi::TypeList((YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (-2)].yystate.yysemantics.yysval.valType), (YY_CAST (yyGLRStackItem const *, yyvsp)[YYFILL (0)].yystate.yysemantics.yysval.valType), make_loc((*yylocp))); }
#line 2674 "bi/parser.cpp"
    break;

  case 268:
#line 791 "bi/parser.ypp"
                 { ((*yyvalp).valType) = empty_type((*yylocp)); }
#line 2680 "bi/parser.cpp"
    break;


#line 2684 "bi/parser.cpp"

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



#line 794 "bi/parser.ypp"

