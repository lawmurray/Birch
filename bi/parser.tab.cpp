/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.0.4"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* Copy the first part of user declarations.  */

#line 67 "parser.tab.cpp" /* yacc.c:339  */

# ifndef YY_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULLPTR nullptr
#  else
#   define YY_NULLPTR 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif


/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif
/* "%code requires" blocks.  */
#line 1 "parser.ypp" /* yacc.c:355  */

  #include "lexer.hpp"
  #include "build/Compiler.hpp"

  extern bi::Compiler* compiler;
  extern char *yytext;

#line 102 "parser.tab.cpp" /* yacc.c:355  */

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    IMPORT = 258,
    PROG = 259,
    MODEL = 260,
    FUNC = 261,
    IF = 262,
    ELSE = 263,
    WHILE = 264,
    CPP = 265,
    HPP = 266,
    THIS = 267,
    DOUBLE_BRACE_OPEN = 268,
    DOUBLE_BRACE_CLOSE = 269,
    RAW = 270,
    NAME = 271,
    BOOL_LITERAL = 272,
    INT_LITERAL = 273,
    REAL_LITERAL = 274,
    STRING_LITERAL = 275,
    LEFT_OP = 276,
    RIGHT_OP = 277,
    LEFT_TILDE_OP = 278,
    RIGHT_TILDE_OP = 279,
    AND_OP = 280,
    OR_OP = 281,
    LE_OP = 282,
    GE_OP = 283,
    EQ_OP = 284,
    NE_OP = 285,
    RANGE_OP = 286
  };
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 37 "parser.ypp" /* yacc.c:355  */

  bool valBool;
  int32_t valInt;
  double valReal;
  const char* valString;

  bi::Name* valName;
  bi::Path* valPath;
  bi::Prog* valProg;
  bi::Expression* valExpression;
  bi::Type* valType;
  bi::Statement* valStatement;

#line 160 "parser.tab.cpp" /* yacc.c:355  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif

/* Location type.  */
#if ! defined YYLTYPE && ! defined YYLTYPE_IS_DECLARED
typedef struct YYLTYPE YYLTYPE;
struct YYLTYPE
{
  int first_line;
  int first_column;
  int last_line;
  int last_column;
};
# define YYLTYPE_IS_DECLARED 1
# define YYLTYPE_IS_TRIVIAL 1
#endif


extern YYSTYPE yylval;
extern YYLTYPE yylloc;
int yyparse (void);



/* Copy the second part of user declarations.  */

#line 191 "parser.tab.cpp" /* yacc.c:358  */
/* Unqualified %code blocks.  */
#line 9 "parser.ypp" /* yacc.c:359  */

  #include "expression/all.hpp"
  #include "program/all.hpp"
  #include "statement/all.hpp"
  #include "type/all.hpp"

  #include <sstream>

  std::stringstream raw;
  
  void setloc(bi::Located* o, YYLTYPE& loc) {
    o->loc->file = compiler->file;
    o->loc->firstLine = loc.first_line;
    o->loc->lastLine = loc.last_line;
    o->loc->firstCol = loc.first_column;
    o->loc->lastCol = loc.last_column;
  }

  bi::Location* make_loc(YYLTYPE& loc) {
    return new bi::Location(compiler->file, loc.first_line, loc.last_line,
        loc.first_column, loc.last_column);
  }
  
  bi::Expression* make_empty() {
    return new bi::EmptyExpression();
  }

#line 221 "parser.tab.cpp" /* yacc.c:359  */

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

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


#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL \
             && defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
  YYLTYPE yyls_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE) + sizeof (YYLTYPE)) \
      + 2 * YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYSIZE_T yynewbytes;                                            \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / sizeof (*yyptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  50
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   284

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  51
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  101
/* YYNRULES -- Number of rules.  */
#define YYNRULES  190
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  274

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   286

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    44,     2,     2,     2,     2,     2,     2,
      39,    40,    48,    46,    43,    47,    32,    49,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    38,    37,
      41,    42,    50,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    33,     2,    34,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    35,     2,    36,    45,     2,     2,     2,
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
      25,    26,    27,    28,    29,    30,    31
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   117,   117,   121,   122,   123,   124,   128,   129,   138,
     142,   146,   150,   154,   155,   156,   157,   166,   170,   174,
     178,   179,   183,   187,   188,   192,   196,   197,   206,   207,
     211,   211,   220,   221,   225,   226,   227,   231,   232,   233,
     237,   241,   245,   246,   250,   251,   255,   256,   260,   261,
     265,   269,   270,   274,   275,   279,   280,   289,   293,   297,
     301,   302,   306,   307,   311,   312,   321,   325,   326,   330,
     331,   335,   336,   340,   341,   345,   346,   350,   355,   364,
     365,   369,   373,   377,   381,   382,   383,   384,   385,   389,
     390,   394,   395,   399,   400,   404,   408,   409,   413,   414,
     415,   419,   420,   424,   425,   429,   430,   434,   435,   439,
     440,   444,   445,   446,   447,   451,   452,   456,   457,   461,
     462,   466,   470,   471,   475,   479,   480,   484,   485,   486,
     487,   491,   492,   496,   500,   501,   502,   503,   504,   505,
     506,   515,   519,   523,   527,   531,   535,   536,   537,   541,
     545,   549,   553,   554,   555,   556,   557,   561,   562,   566,
     567,   571,   575,   576,   580,   581,   585,   586,   587,   591,
     592,   596,   597,   601,   605,   606,   610,   611,   615,   619,
     620,   621,   622,   623,   624,   625,   629,   630,   634,   635,
     639
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "IMPORT", "PROG", "MODEL", "FUNC", "IF",
  "ELSE", "WHILE", "CPP", "HPP", "THIS", "DOUBLE_BRACE_OPEN",
  "DOUBLE_BRACE_CLOSE", "RAW", "NAME", "BOOL_LITERAL", "INT_LITERAL",
  "REAL_LITERAL", "STRING_LITERAL", "LEFT_OP", "RIGHT_OP", "LEFT_TILDE_OP",
  "RIGHT_TILDE_OP", "AND_OP", "OR_OP", "LE_OP", "GE_OP", "EQ_OP", "NE_OP",
  "RANGE_OP", "'.'", "'['", "']'", "'{'", "'}'", "';'", "':'", "'('",
  "')'", "'<'", "'='", "','", "'!'", "'~'", "'+'", "'-'", "'*'", "'/'",
  "'>'", "$accept", "name", "path_name", "path", "bool_literal",
  "int_literal", "real_literal", "string_literal", "literal", "brackets",
  "braces", "func_braces", "optional_func_braces", "model_braces",
  "optional_model_braces", "prog_braces", "optional_prog_braces", "raw",
  "double_braces", "$@1", "var_parameter", "func_parameter",
  "model_parameter", "prog_parameter", "value", "optional_value",
  "parameter_list", "parameters", "optional_parameters", "result",
  "result_list", "results", "optional_results", "var_reference",
  "func_reference", "model_reference", "argument_list", "arguments",
  "optional_arguments", "parens_type", "primary_type", "brackets_type",
  "lambda_type", "assignable_type", "random_type", "list_type", "type",
  "reference_expression", "parens_expression", "lambda_expression",
  "this_expression", "primary_expression", "index_expression",
  "index_list", "brackets_expression", "member_operator",
  "member_expression", "unary_operator", "unary_expression",
  "multiplicative_operator", "multiplicative_expression",
  "additive_operator", "additive_expression", "relational_operator",
  "relational_expression", "equality_operator", "equality_expression",
  "logical_and_operator", "logical_and_expression", "logical_or_operator",
  "logical_or_expression", "assignment_operator", "assignment_expression",
  "expression", "binary_operator", "var_declaration", "func_declaration",
  "model_declaration", "prog_declaration", "expression_statement", "if",
  "while", "cpp", "hpp", "statement", "statements", "optional_statements",
  "func_statement", "func_statements", "optional_func_statements",
  "model_statement", "model_statements", "optional_model_statements",
  "prog_statement", "prog_statements", "optional_prog_statements",
  "import", "file_statement", "file_statements",
  "optional_file_statements", "file", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,    46,    91,    93,   123,   125,    59,    58,    40,
      41,    60,    61,    44,    33,   126,    43,    45,    42,    47,
      62
};
# endif

#define YYPACT_NINF -193

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-193)))

#define YYTABLE_NINF -1

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
      40,   104,    12,    12,     3,     8,     8,  -193,    21,    25,
      31,  -193,  -193,  -193,  -193,  -193,  -193,  -193,    40,  -193,
    -193,    80,  -193,  -193,  -193,  -193,    75,    86,   100,  -193,
      72,  -193,    58,   100,  -193,  -193,  -193,  -193,    21,   111,
    -193,    95,    96,  -193,   108,  -193,  -193,    21,  -193,  -193,
    -193,   104,  -193,    43,    -1,    12,  -193,    -8,  -193,  -193,
    -193,   213,    38,   151,   148,   138,  -193,   152,   170,  -193,
    -193,    21,   111,  -193,  -193,   160,  -193,  -193,  -193,    85,
     125,  -193,  -193,  -193,   154,    19,  -193,    12,  -193,  -193,
    -193,  -193,  -193,  -193,  -193,  -193,  -193,  -193,  -193,  -193,
    -193,  -193,  -193,  -193,  -193,  -193,  -193,  -193,  -193,  -193,
    -193,  -193,  -193,    38,   153,     1,    81,  -193,    78,    21,
    -193,  -193,  -193,  -193,  -193,  -193,   158,  -193,   111,  -193,
    -193,  -193,  -193,  -193,   151,  -193,  -193,   155,  -193,  -193,
    -193,  -193,   161,  -193,   167,   141,  -193,    98,   102,    71,
     136,   175,    49,  -193,   163,  -193,  -193,   169,   206,  -193,
    -193,  -193,    38,   162,   168,   171,  -193,  -193,  -193,  -193,
    -193,  -193,  -193,   125,   176,  -193,  -193,  -193,  -193,    19,
    -193,   177,   103,   179,   151,    38,  -193,  -193,  -193,   125,
    -193,  -193,  -193,  -193,  -193,  -193,   129,   181,  -193,    81,
    -193,   206,  -193,   141,   167,   206,   206,   206,   206,   206,
     206,   206,   206,  -193,  -193,  -193,   206,   206,  -193,  -193,
    -193,  -193,  -193,  -193,   151,    81,  -193,    90,  -193,   125,
    -193,   191,  -193,  -193,   172,   194,   180,  -193,  -193,    98,
     102,    71,   136,   175,  -193,  -193,   189,   190,    81,  -193,
    -193,    38,  -193,  -193,   206,  -193,   206,   196,   196,  -193,
    -193,  -193,  -193,   125,   224,  -193,   125,  -193,   197,    13,
    -193,  -193,  -193,  -193
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
     189,     0,     0,     0,     0,     0,     0,     2,     0,     0,
       0,   182,   183,   185,   184,   180,   181,   179,   186,   188,
     190,     0,     6,     5,     4,     3,     7,     0,     0,   144,
      49,   143,     0,     0,   142,    30,   150,   151,     0,    65,
      67,    68,    69,    71,    73,    78,    33,     0,   141,   187,
       1,     0,   178,     0,     0,     0,    48,     0,   100,    98,
      99,     0,     0,    56,     0,    75,    77,     0,     0,    64,
      59,     0,     0,    70,    74,    43,     8,    46,    44,     0,
     177,    27,    26,    40,     0,   172,    24,     0,    23,    38,
     127,   128,   129,   121,   124,   113,   114,   117,   118,   111,
     130,   107,   108,   103,   104,   112,   134,   135,   136,   137,
     138,   139,   140,     0,     0,     0,     0,    28,     0,     0,
      66,    83,     9,    10,    11,    12,     0,    62,    57,    13,
      14,    15,    16,    84,    56,    79,    80,     0,    87,    85,
      86,    88,    93,    96,   101,     0,   105,   109,   115,   119,
     122,   125,   131,   133,    60,    68,    72,     0,     0,    42,
      32,    47,     0,     0,     0,     0,   152,   153,   154,   155,
     156,   173,   174,   176,     0,    39,   166,   167,   168,   169,
     171,     0,     0,     0,    56,     0,    50,    53,    55,   165,
      21,    20,    34,    31,    29,    76,    57,     0,    58,     0,
      63,     0,    95,     0,   102,     0,     0,     0,     0,     0,
       0,     0,     0,    17,    41,    45,     0,     0,   145,   175,
      25,   170,    22,    37,    56,     0,    51,     0,   161,   162,
     164,     0,    81,    82,    91,     0,    90,    97,   106,   110,
     116,   120,   123,   126,   132,    61,     0,     0,     0,    35,
      54,     0,   163,    19,     0,    94,     0,     0,     0,    36,
      52,    92,    89,   160,   148,   149,   157,   159,     0,     0,
     158,    18,   146,   147
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -193,     0,  -193,   184,  -193,  -193,  -193,  -193,  -193,  -193,
    -192,  -193,  -184,  -193,    62,  -193,  -193,  -193,   240,  -193,
     -31,  -193,  -193,  -193,  -193,  -193,  -193,    94,  -193,  -172,
    -193,  -193,  -123,  -193,  -193,   -29,    35,   -23,  -193,   178,
    -193,   185,  -193,   -28,  -193,  -193,   201,  -193,  -193,  -193,
    -193,  -193,  -193,    10,    52,  -193,   112,   233,    61,   207,
      63,   209,    60,   210,    64,   212,    65,   214,    66,   126,
    -193,   127,  -187,   -61,  -193,   -79,   -76,  -193,  -193,  -193,
      11,  -193,    88,     5,  -177,    15,  -193,  -193,    48,  -193,
    -193,   105,  -193,   109,  -193,  -193,  -193,  -193,   265,  -193,
    -193
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,   128,    26,    27,   129,   130,   131,   132,   133,    73,
     264,   191,   192,    88,    89,    82,    83,   118,    36,    64,
      10,    34,    31,    29,   159,   160,    79,   134,    57,   187,
     227,   188,   116,   135,   136,    40,   137,   198,    70,    41,
      42,    43,    44,    45,    66,    67,    46,   138,   139,   140,
     141,   142,   234,   235,   143,   203,   144,   145,   146,   205,
     147,   206,   148,   207,   149,   208,   150,   209,   151,   111,
     152,   112,   153,   165,   113,    11,    12,    13,    14,   166,
     167,   168,   169,   170,   171,   267,   268,   229,   230,   231,
     179,   180,   181,   172,   173,   174,    17,    18,    19,    20,
      21
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint16 yytable[] =
{
       9,    61,    28,    30,    33,    16,   176,   154,    39,   177,
      65,   199,   228,   226,   236,   233,    69,     7,     9,     7,
     163,    35,    78,    16,   244,     4,    84,    85,     7,    86,
       6,   114,     9,    87,    80,     7,    81,     7,    39,     8,
     185,   249,    32,     1,     2,     3,     4,    39,   263,   157,
       5,     6,   228,     9,     7,    39,     7,     8,   182,     7,
      38,   225,     9,    47,   259,   197,   265,   236,    48,   262,
      90,    39,    91,    92,     7,    94,     8,   272,     8,   260,
      50,     8,   183,    77,   186,     9,   266,    39,    15,   266,
     178,   195,   193,   194,   100,    78,     8,   214,    95,    96,
     176,   248,    58,   177,    59,    60,    15,    51,    22,    23,
      24,    53,    99,     9,    55,     9,   189,    71,   190,    39,
      25,   105,    54,    52,    56,   161,   196,    63,   162,    72,
     250,   215,   163,   251,   164,     5,     6,   121,    85,    53,
      86,     7,   122,   123,   124,   125,   103,   104,   101,   102,
      68,   154,    74,   121,   186,   246,   247,     7,   122,   123,
     124,   125,     9,   117,   126,    97,    98,    47,    68,    58,
     121,    59,    60,   115,     7,   122,   123,   124,   125,     9,
     126,   158,   121,   119,   178,     9,     7,   122,   123,   124,
     125,   175,   120,   184,   201,   200,     8,   126,    77,   202,
      93,   216,    58,   213,    59,    60,   212,   217,   218,   126,
     127,   256,   220,   222,    58,   254,    59,    60,   121,   224,
     186,   232,     7,   122,   123,   124,   125,   253,   255,   257,
     258,   263,   269,   271,    90,    76,    91,    92,    93,    94,
      95,    96,    97,    98,   223,   126,    37,   245,    75,   155,
      58,     9,    59,    60,    99,   237,   156,   204,   100,   101,
     102,   103,   104,   105,   261,    62,   238,   240,   106,   239,
     107,   108,   241,   109,   242,   110,   243,   252,   210,   211,
     273,   270,   219,    49,   221
};

static const yytype_uint16 yycheck[] =
{
       0,    32,     2,     3,     4,     0,    85,    68,     8,    85,
      38,   134,   189,   185,   201,   199,    39,    16,    18,    16,
       7,    13,    53,    18,   211,     6,    55,    35,    16,    37,
      11,    62,    32,    41,    35,    16,    37,    16,    38,    38,
      39,   225,    39,     3,     4,     5,     6,    47,    35,    72,
      10,    11,   229,    53,    16,    55,    16,    38,    87,    16,
      39,   184,    62,    38,   248,   126,   258,   254,    37,   256,
      21,    71,    23,    24,    16,    26,    38,   269,    38,   251,
       0,    38,   113,    40,   115,    85,   263,    87,     0,   266,
      85,   119,    14,    15,    45,   126,    38,   158,    27,    28,
     179,   224,    44,   179,    46,    47,    18,    32,     4,     5,
       6,    39,    41,   113,    42,   115,    35,    22,    37,   119,
      16,    50,    28,    37,    30,    40,   126,    33,    43,    33,
      40,   162,     7,    43,     9,    10,    11,    12,    35,    39,
      37,    16,    17,    18,    19,    20,    48,    49,    46,    47,
      39,   212,    44,    12,   185,   216,   217,    16,    17,    18,
      19,    20,   162,    15,    39,    29,    30,    38,    39,    44,
      12,    46,    47,    22,    16,    17,    18,    19,    20,   179,
      39,    21,    12,    45,   179,   185,    16,    17,    18,    19,
      20,    37,    40,    40,    33,    40,    38,    39,    40,    32,
      25,    39,    44,    34,    46,    47,    43,    39,    37,    39,
      40,    31,    36,    36,    44,    43,    46,    47,    12,    40,
     251,    40,    16,    17,    18,    19,    20,    36,    34,    40,
      40,    35,     8,    36,    21,    51,    23,    24,    25,    26,
      27,    28,    29,    30,   182,    39,     6,   212,    47,    71,
      44,   251,    46,    47,    41,   203,    71,   145,    45,    46,
      47,    48,    49,    50,   254,    32,   205,   207,    61,   206,
      61,    61,   208,    61,   209,    61,   210,   229,   152,   152,
     269,   266,   173,    18,   179
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,     5,     6,    10,    11,    16,    38,    52,
      71,   126,   127,   128,   129,   133,   134,   147,   148,   149,
     150,   151,     4,     5,     6,    16,    53,    54,    52,    74,
      52,    73,    39,    52,    72,    13,    69,    69,    39,    52,
      86,    90,    91,    92,    93,    94,    97,    38,    37,   149,
       0,    32,    37,    39,    78,    42,    78,    79,    44,    46,
      47,    71,   108,    78,    70,    94,    95,    96,    39,    88,
      89,    22,    33,    60,    44,    97,    54,    40,    71,    77,
      35,    37,    66,    67,    86,    35,    37,    41,    64,    65,
      21,    23,    24,    25,    26,    27,    28,    29,    30,    41,
      45,    46,    47,    48,    49,    50,   110,   112,   114,   116,
     118,   120,   122,   125,    71,    22,    83,    15,    68,    45,
      40,    12,    17,    18,    19,    20,    39,    40,    52,    55,
      56,    57,    58,    59,    78,    84,    85,    87,    98,    99,
     100,   101,   102,   105,   107,   108,   109,   111,   113,   115,
     117,   119,   121,   123,   124,    90,    92,    88,    21,    75,
      76,    40,    43,     7,     9,   124,   130,   131,   132,   133,
     134,   135,   144,   145,   146,    37,   126,   127,   134,   141,
     142,   143,    86,    71,    40,    39,    71,    80,    82,    35,
      37,    62,    63,    14,    15,    94,    52,   124,    88,    83,
      40,    33,    32,   106,   107,   110,   112,   114,   116,   118,
     120,   122,    43,    34,   124,    71,    39,    39,    37,   144,
      36,   142,    36,    65,    40,    83,    80,    81,   135,   138,
     139,   140,    40,    63,   103,   104,   123,   105,   109,   111,
     113,   115,   117,   119,   123,    87,   124,   124,    83,    63,
      40,    43,   139,    36,    43,    34,    31,    40,    40,    63,
      80,   104,   123,    35,    61,    61,   135,   136,   137,     8,
     136,    36,    61,   131
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    51,    52,    53,    53,    53,    53,    54,    54,    55,
      56,    57,    58,    59,    59,    59,    59,    60,    61,    62,
      63,    63,    64,    65,    65,    66,    67,    67,    68,    68,
      70,    69,    71,    71,    72,    72,    72,    73,    73,    73,
      74,    75,    76,    76,    77,    77,    78,    78,    79,    79,
      80,    81,    81,    82,    82,    83,    83,    84,    85,    86,
      87,    87,    88,    88,    89,    89,    90,    91,    91,    92,
      92,    93,    93,    94,    94,    95,    95,    96,    97,    98,
      98,    99,   100,   101,   102,   102,   102,   102,   102,   103,
     103,   104,   104,   105,   105,   106,   107,   107,   108,   108,
     108,   109,   109,   110,   110,   111,   111,   112,   112,   113,
     113,   114,   114,   114,   114,   115,   115,   116,   116,   117,
     117,   118,   119,   119,   120,   121,   121,   122,   122,   122,
     122,   123,   123,   124,   125,   125,   125,   125,   125,   125,
     125,   126,   127,   128,   129,   130,   131,   131,   131,   132,
     133,   134,   135,   135,   135,   135,   135,   136,   136,   137,
     137,   138,   139,   139,   140,   140,   141,   141,   141,   142,
     142,   143,   143,   144,   145,   145,   146,   146,   147,   148,
     148,   148,   148,   148,   148,   148,   149,   149,   150,   150,
     151
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     1,     1,     1,     1,     1,     3,     1,
       1,     1,     1,     1,     1,     1,     1,     3,     3,     3,
       1,     1,     3,     1,     1,     3,     1,     1,     1,     2,
       0,     4,     4,     2,     4,     6,     7,     5,     3,     4,
       3,     2,     1,     0,     1,     3,     2,     3,     1,     0,
       1,     1,     3,     1,     3,     2,     0,     1,     2,     2,
       1,     3,     2,     3,     1,     0,     3,     1,     1,     1,
       2,     1,     3,     1,     2,     1,     3,     1,     1,     1,
       1,     3,     3,     1,     1,     1,     1,     1,     1,     3,
       1,     1,     3,     1,     4,     1,     1,     3,     1,     1,
       1,     1,     2,     1,     1,     1,     3,     1,     1,     1,
       3,     1,     1,     1,     1,     1,     3,     1,     1,     1,
       3,     1,     1,     3,     1,     1,     3,     1,     1,     1,
       1,     1,     3,     1,     1,     1,     1,     1,     1,     1,
       1,     2,     2,     2,     2,     2,     7,     7,     5,     5,
       2,     2,     1,     1,     1,     1,     1,     1,     2,     1,
       0,     1,     1,     2,     1,     0,     1,     1,     1,     1,
       2,     1,     0,     1,     1,     2,     1,     0,     3,     1,
       1,     1,     1,     1,     1,     1,     1,     2,     1,     0,
       1
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                  \
do                                                              \
  if (yychar == YYEMPTY)                                        \
    {                                                           \
      yychar = (Token);                                         \
      yylval = (Value);                                         \
      YYPOPSTACK (yylen);                                       \
      yystate = *yyssp;                                         \
      goto yybackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;                                                  \
    }                                                           \
while (0)

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256


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

#define YYRHSLOC(Rhs, K) ((Rhs)[K])


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)


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


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value, Location); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
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
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
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

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yytype_int16 *yyssp, YYSTYPE *yyvsp, YYLTYPE *yylsp, int yyrule)
{
  unsigned long int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[yyssp[yyi + 1 - yynrhs]],
                       &(yyvsp[(yyi + 1) - (yynrhs)])
                       , &(yylsp[(yyi + 1) - (yynrhs)])                       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, yylsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
yystrlen (const char *yystr)
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

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
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
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
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
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
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
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
                yyarg[yycount++] = yytname[yyx];
                {
                  YYSIZE_T yysize1 = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
                  if (! (yysize <= yysize1
                         && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                    return 2;
                  yysize = yysize1;
                }
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    YYSIZE_T yysize1 = yysize + yystrlen (yyformat);
    if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
      return 2;
    yysize = yysize1;
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
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
  return 0;
}
#endif /* YYERROR_VERBOSE */

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




/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Location data for the lookahead symbol.  */
YYLTYPE yylloc
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
  = { 1, 1, 1, 1 }
# endif
;
/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.
       'yyls': related to locations.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    /* The location stack.  */
    YYLTYPE yylsa[YYINITDEPTH];
    YYLTYPE *yyls;
    YYLTYPE *yylsp;

    /* The locations where the error started and ended.  */
    YYLTYPE yyerror_range[3];

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;
  YYLTYPE yyloc;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N), yylsp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yylsp = yyls = yylsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  yylsp[0] = yylloc;
  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        YYSTYPE *yyvs1 = yyvs;
        yytype_int16 *yyss1 = yyss;
        YYLTYPE *yyls1 = yyls;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * sizeof (*yyssp),
                    &yyvs1, yysize * sizeof (*yyvsp),
                    &yyls1, yysize * sizeof (*yylsp),
                    &yystacksize);

        yyls = yyls1;
        yyss = yyss1;
        yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yytype_int16 *yyss1 = yyss;
        union yyalloc *yyptr =
          (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
        YYSTACK_RELOCATE (yyls_alloc, yyls);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;
      yylsp = yyls + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
                  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
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

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END
  *++yylsp = yylloc;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];

  /* Default location.  */
  YYLLOC_DEFAULT (yyloc, (yylsp - yylen), yylen);
  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 117 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name((yyvsp[0].valString), make_loc((yyloc))); }
#line 1636 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 3:
#line 121 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name((yyvsp[0].valString), make_loc((yyloc))); }
#line 1642 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 4:
#line 122 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("function", make_loc((yyloc))); }
#line 1648 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 5:
#line 123 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("model", make_loc((yyloc))); }
#line 1654 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 6:
#line 124 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("program", make_loc((yyloc))); }
#line 1660 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 7:
#line 128 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valPath) = new bi::Path((yyvsp[0].valName), nullptr, make_loc((yyloc))); }
#line 1666 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 8:
#line 129 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valPath) = new bi::Path((yyvsp[-2].valName), (yyvsp[0].valPath), make_loc((yyloc))); }
#line 1672 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 9:
#line 138 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BooleanLiteral((yyvsp[0].valBool), yytext, new bi::ModelReference(new bi::Name("Boolean")), make_loc((yyloc))); }
#line 1678 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 10:
#line 142 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::IntegerLiteral((yyvsp[0].valInt), yytext, new bi::ModelReference(new bi::Name("Integer")), make_loc((yyloc))); }
#line 1684 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 11:
#line 146 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::RealLiteral((yyvsp[0].valReal), yytext, new bi::ModelReference(new bi::Name("Real")), make_loc((yyloc))); }
#line 1690 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 12:
#line 150 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::StringLiteral((yyvsp[0].valString), yytext, new bi::ModelReference(new bi::Name("String")), make_loc((yyloc))); }
#line 1696 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 17:
#line 166 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = (yyvsp[-1].valExpression); }
#line 1702 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 18:
#line 170 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracesExpression((yyvsp[-1].valStatement), make_loc((yyloc))); }
#line 1708 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 19:
#line 174 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracesExpression((yyvsp[-1].valStatement), make_loc((yyloc))); }
#line 1714 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 21:
#line 179 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_empty(); }
#line 1720 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 22:
#line 183 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracesExpression((yyvsp[-1].valStatement), make_loc((yyloc))); }
#line 1726 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 24:
#line 188 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_empty(); }
#line 1732 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 25:
#line 192 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracesExpression((yyvsp[-1].valStatement), make_loc((yyloc))); }
#line 1738 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 27:
#line 197 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_empty(); }
#line 1744 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 28:
#line 206 "parser.ypp" /* yacc.c:1661  */
    { raw << (yyvsp[0].valString); }
#line 1750 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 29:
#line 207 "parser.ypp" /* yacc.c:1661  */
    { raw << (yyvsp[0].valString); }
#line 1756 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 30:
#line 211 "parser.ypp" /* yacc.c:1661  */
    { raw.str(""); }
#line 1762 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 32:
#line 220 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::VarParameter((yyvsp[-3].valName), (yyvsp[-1].valType), make_empty(), make_empty(), make_loc((yyloc))); }
#line 1768 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 33:
#line 221 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::VarParameter(new bi::Name(), (yyvsp[0].valType), make_empty(), make_empty(), make_loc((yyloc))); }
#line 1774 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 34:
#line 225 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::FuncParameter((yyvsp[-3].valName), (yyvsp[-2].valExpression), (yyvsp[-1].valExpression), (yyvsp[0].valExpression), bi::FUNCTION, make_loc((yyloc))); }
#line 1780 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 35:
#line 226 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::FuncParameter((yyvsp[-4].valName), (yyvsp[-3].valExpression), (yyvsp[-1].valExpression), (yyvsp[0].valExpression), bi::UNARY, make_loc((yyloc))); }
#line 1786 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 36:
#line 227 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::FuncParameter((yyvsp[-5].valExpression), (yyvsp[-4].valName), (yyvsp[-3].valExpression), (yyvsp[-1].valExpression), (yyvsp[0].valExpression), bi::BINARY, make_loc((yyloc))); }
#line 1792 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 37:
#line 231 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-4].valName), (yyvsp[-3].valExpression), new bi::Name("<"), (yyvsp[-1].valType), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1798 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 38:
#line 232 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-2].valName), (yyvsp[-1].valExpression), new bi::Name(), new bi::EmptyType(), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1804 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 39:
#line 233 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-3].valName), make_empty(), new bi::Name("="), (yyvsp[-1].valType), make_empty(), make_loc((yyloc))); }
#line 1810 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 40:
#line 237 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valProg) = new bi::ProgParameter((yyvsp[-2].valName), (yyvsp[-1].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1816 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 41:
#line 241 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = (yyvsp[0].valExpression); }
#line 1822 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 43:
#line 246 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_empty(); }
#line 1828 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 45:
#line 251 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ExpressionList((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1834 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 46:
#line 255 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_empty(); }
#line 1840 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 47:
#line 256 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = (yyvsp[-1].valExpression); }
#line 1846 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 49:
#line 261 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_empty(); }
#line 1852 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 52:
#line 270 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ExpressionList((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1858 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 54:
#line 275 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = (yyvsp[-1].valExpression); }
#line 1864 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 55:
#line 279 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = (yyvsp[0].valExpression); }
#line 1870 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 56:
#line 280 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_empty(); }
#line 1876 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 57:
#line 289 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::VarReference((yyvsp[0].valName), make_loc((yyloc))); }
#line 1882 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 58:
#line 293 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::FuncReference((yyvsp[-1].valName), (yyvsp[0].valExpression), bi::FUNCTION, make_loc((yyloc))); }
#line 1888 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 59:
#line 297 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelReference((yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1894 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 61:
#line 302 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ExpressionList((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1900 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 62:
#line 306 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_empty(); }
#line 1906 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 63:
#line 307 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = (yyvsp[-1].valExpression); }
#line 1912 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 65:
#line 312 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_empty(); }
#line 1918 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 66:
#line 321 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ParenthesesType((yyvsp[-1].valType), make_loc((yyloc))); }
#line 1924 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 70:
#line 331 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::BracketsType((yyvsp[-1].valType), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1930 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 72:
#line 336 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::LambdaType((yyvsp[-2].valType), (yyvsp[0].valType), make_loc((yyloc))); }
#line 1936 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 74:
#line 341 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::AssignableType((yyvsp[-1].valType), make_loc((yyloc))); }
#line 1942 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 76:
#line 346 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::RandomType((yyvsp[-2].valType), (yyvsp[0].valType), make_loc((yyloc))); }
#line 1948 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 81:
#line 369 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ParenthesesExpression((yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 1954 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 82:
#line 373 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::FuncParameter(new bi::Name(), (yyvsp[-2].valExpression), (yyvsp[-1].valExpression), (yyvsp[0].valExpression), bi::LAMBDA, make_loc((yyloc))); }
#line 1960 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 83:
#line 377 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::This(make_loc((yyloc))); }
#line 1966 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 89:
#line 389 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::Range((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1972 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 90:
#line 390 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::Index((yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1978 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 92:
#line 395 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ExpressionList((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1984 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 94:
#line 400 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracketsExpression((yyvsp[-3].valExpression), (yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 1990 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 97:
#line 409 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::Member((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1996 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 98:
#line 413 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("+", make_loc((yyloc))); }
#line 2002 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 99:
#line 414 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("-", make_loc((yyloc))); }
#line 2008 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 100:
#line 415 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("!", make_loc((yyloc))); }
#line 2014 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 102:
#line 420 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::FuncReference((yyvsp[-1].valName), (yyvsp[0].valExpression), bi::UNARY, make_loc((yyloc))); }
#line 2020 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 103:
#line 424 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("*", make_loc((yyloc))); }
#line 2026 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 104:
#line 425 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("/", make_loc((yyloc))); }
#line 2032 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 106:
#line 430 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::FuncReference((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), bi::BINARY, make_loc((yyloc))); }
#line 2038 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 107:
#line 434 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("+", make_loc((yyloc))); }
#line 2044 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 108:
#line 435 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("-", make_loc((yyloc))); }
#line 2050 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 110:
#line 440 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::FuncReference((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), bi::BINARY, make_loc((yyloc))); }
#line 2056 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 111:
#line 444 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("<", make_loc((yyloc))); }
#line 2062 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 112:
#line 445 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name(">", make_loc((yyloc))); }
#line 2068 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 113:
#line 446 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("<=", make_loc((yyloc))); }
#line 2074 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 114:
#line 447 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name(">=", make_loc((yyloc))); }
#line 2080 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 116:
#line 452 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::FuncReference((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), bi::BINARY, make_loc((yyloc))); }
#line 2086 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 117:
#line 456 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("==", make_loc((yyloc))); }
#line 2092 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 118:
#line 457 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("!=", make_loc((yyloc))); }
#line 2098 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 120:
#line 462 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::FuncReference((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), bi::BINARY, make_loc((yyloc))); }
#line 2104 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 121:
#line 466 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("&&", make_loc((yyloc))); }
#line 2110 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 123:
#line 471 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::FuncReference((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), bi::BINARY, make_loc((yyloc))); }
#line 2116 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 124:
#line 475 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("||", make_loc((yyloc))); }
#line 2122 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 126:
#line 480 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::FuncReference((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), bi::BINARY, make_loc((yyloc))); }
#line 2128 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 127:
#line 484 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("<-", make_loc((yyloc))); }
#line 2134 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 128:
#line 485 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("<~", make_loc((yyloc))); }
#line 2140 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 129:
#line 486 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("~>", make_loc((yyloc))); }
#line 2146 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 130:
#line 487 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("~", make_loc((yyloc))); }
#line 2152 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 132:
#line 492 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::FuncReference((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), bi::ASSIGN, make_loc((yyloc))); }
#line 2158 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 141:
#line 515 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::VarDeclaration(dynamic_cast<bi::VarParameter*>((yyvsp[-1].valExpression)), make_loc((yyloc))); }
#line 2164 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 142:
#line 519 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::FuncDeclaration(dynamic_cast<bi::FuncParameter*>((yyvsp[0].valExpression)), make_loc((yyloc))); }
#line 2170 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 143:
#line 523 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::ModelDeclaration(dynamic_cast<bi::ModelParameter*>((yyvsp[0].valType)), make_loc((yyloc))); }
#line 2176 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 144:
#line 527 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::ProgDeclaration(dynamic_cast<bi::ProgParameter*>((yyvsp[0].valProg)), make_loc((yyloc))); }
#line 2182 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 145:
#line 531 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::ExpressionStatement((yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 2188 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 146:
#line 535 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Conditional((yyvsp[-4].valExpression), (yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2194 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 147:
#line 536 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Conditional((yyvsp[-4].valExpression), (yyvsp[-2].valExpression), new bi::BracesExpression((yyvsp[0].valStatement)), make_loc((yyloc))); }
#line 2200 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 148:
#line 537 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Conditional((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_empty(), make_loc((yyloc))); }
#line 2206 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 149:
#line 541 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Loop((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2212 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 150:
#line 545 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Raw(new bi::Name("cpp"), raw.str(), make_loc((yyloc))); }
#line 2218 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 151:
#line 549 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Raw(new bi::Name("hpp"), raw.str(), make_loc((yyloc))); }
#line 2224 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 158:
#line 562 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2230 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 160:
#line 567 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2236 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 163:
#line 576 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2242 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 165:
#line 581 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2248 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 170:
#line 592 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2254 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 172:
#line 597 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2260 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 175:
#line 606 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2266 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 177:
#line 611 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2272 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 178:
#line 615 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Import((yyvsp[-1].valPath), compiler->import((yyvsp[-1].valPath)), make_loc((yyloc))); }
#line 2278 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 187:
#line 630 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2284 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 189:
#line 635 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2290 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 190:
#line 639 "parser.ypp" /* yacc.c:1661  */
    { compiler->setRoot((yyvsp[0].valStatement)); }
#line 2296 "parser.tab.cpp" /* yacc.c:1661  */
    break;


#line 2300 "parser.tab.cpp" /* yacc.c:1661  */
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;
  *++yylsp = yyloc;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }

  yyerror_range[1] = yylloc;

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval, &yylloc);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  yyerror_range[1] = yylsp[1-yylen];
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYTERROR;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;

      yyerror_range[1] = *yylsp;
      yydestruct ("Error: popping",
                  yystos[yystate], yyvsp, yylsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  yyerror_range[2] = yylloc;
  /* Using YYLLOC is tempting, but would change the location of
     the lookahead.  YYLOC is available though.  */
  YYLLOC_DEFAULT (yyloc, yyerror_range, 2);
  *++yylsp = yyloc;

  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval, &yylloc);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  yystos[*yyssp], yyvsp, yylsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  return yyresult;
}
#line 642 "parser.ypp" /* yacc.c:1906  */

