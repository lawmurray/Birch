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
    ANY = 268,
    DOUBLE_BRACE_OPEN = 269,
    DOUBLE_BRACE_CLOSE = 270,
    RAW = 271,
    NAME = 272,
    BOOL_LITERAL = 273,
    INT_LITERAL = 274,
    REAL_LITERAL = 275,
    STRING_LITERAL = 276,
    RIGHT_ARROW = 277,
    LEFT_ARROW = 278,
    RIGHT_TILDE_ARROW = 279,
    LEFT_TILDE_ARROW = 280,
    AND_OP = 281,
    OR_OP = 282,
    LE_OP = 283,
    GE_OP = 284,
    EQ_OP = 285,
    NE_OP = 286,
    RANGE = 287,
    POW_OP = 288
  };
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 55 "parser.ypp" /* yacc.c:355  */

  bool valBool;
  int32_t valInt;
  double valReal;
  char* valString;

  bi::Operator valOperator;
  bi::Name* valName;
  bi::Path* valPath;
  bi::Prog* valProg;
  bi::Expression* valExpression;
  bi::Type* valType;
  bi::Statement* valStatement;

#line 163 "parser.tab.cpp" /* yacc.c:355  */
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

#line 194 "parser.tab.cpp" /* yacc.c:358  */
/* Unqualified %code blocks.  */
#line 9 "parser.ypp" /* yacc.c:359  */

  #include "dimension/all.hpp"
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
  
  bi::Expression* convertReference(bi::Expression* ref,
    bi::Expression* result, bi::Expression* braces, bi::Location* loc) {
    bi::FuncReference* func = dynamic_cast<bi::FuncReference*>(ref);
    bi::BinaryReference* binary = dynamic_cast<bi::BinaryReference*>(ref);
    bi::UnaryReference* unary = dynamic_cast<bi::UnaryReference*>(ref);
    bi::Expression* param;
    
    if (func) {
      param = new bi::FuncParameter(func->name, func->parens.release(), result, braces, loc);
    } else if (binary) {
      param = new bi::BinaryParameter(binary->left.release(), binary->op, binary->right.release(), result, braces, loc);
    } else if (unary) {
      param = new bi::UnaryParameter(unary->op, unary->right.release(), result, braces, loc);
    } else {
      assert(false);
    }
    delete ref;
    
    return param;
  }

#line 242 "parser.tab.cpp" /* yacc.c:359  */

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
#define YYFINAL  33
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   276

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  56
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  107
/* YYNRULES -- Number of rules.  */
#define YYNRULES  204
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  288

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   288

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    47,     2,     2,     2,    51,     2,     2,
      35,    36,    49,    45,    44,    46,    34,    50,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    39,    40,
      52,    55,    53,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    41,     2,    42,    43,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    37,    54,    38,    48,     2,     2,     2,
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
      25,    26,    27,    28,    29,    30,    31,    32,    33
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   137,   137,   141,   142,   143,   144,   148,   149,   158,
     162,   166,   170,   174,   175,   176,   177,   186,   190,   194,
     198,   202,   211,   212,   216,   216,   225,   229,   233,   242,
     246,   247,   248,   252,   253,   254,   255,   256,   257,   261,
     262,   271,   272,   276,   277,   281,   285,   289,   290,   291,
     295,   296,   300,   301,   305,   314,   315,   319,   320,   329,
     330,   334,   338,   342,   343,   344,   345,   349,   350,   354,
     358,   359,   363,   364,   368,   369,   370,   371,   375,   376,
     380,   384,   385,   389,   390,   391,   395,   396,   400,   401,
     405,   406,   410,   414,   415,   419,   423,   424,   428,   429,
     430,   431,   435,   436,   440,   441,   445,   446,   450,   454,
     455,   459,   463,   464,   468,   472,   473,   477,   481,   482,
     486,   487,   491,   492,   496,   500,   501,   510,   511,   515,
     516,   520,   521,   525,   534,   538,   542,   546,   550,   551,
     552,   553,   554,   555,   556,   560,   564,   565,   569,   573,
     577,   578,   579,   583,   587,   591,   595,   596,   597,   598,
     599,   603,   604,   608,   609,   613,   617,   618,   622,   623,
     627,   628,   629,   633,   634,   638,   639,   643,   647,   648,
     652,   653,   657,   661,   662,   663,   664,   668,   669,   673,
     674,   675,   676,   680,   681,   682,   683,   684,   685,   689,
     690,   694,   695,   696,   700
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "IMPORT", "PROG", "MODEL", "FUNC", "IF",
  "ELSE", "WHILE", "CPP", "HPP", "THIS", "ANY", "DOUBLE_BRACE_OPEN",
  "DOUBLE_BRACE_CLOSE", "RAW", "NAME", "BOOL_LITERAL", "INT_LITERAL",
  "REAL_LITERAL", "STRING_LITERAL", "RIGHT_ARROW", "LEFT_ARROW",
  "RIGHT_TILDE_ARROW", "LEFT_TILDE_ARROW", "AND_OP", "OR_OP", "LE_OP",
  "GE_OP", "EQ_OP", "NE_OP", "RANGE", "POW_OP", "'.'", "'('", "')'", "'{'",
  "'}'", "':'", "';'", "'['", "']'", "'^'", "','", "'+'", "'-'", "'!'",
  "'~'", "'*'", "'/'", "'%'", "'<'", "'>'", "'|'", "'='", "$accept",
  "name", "path_name", "path", "bool_literal", "int_literal",
  "real_literal", "string_literal", "literal", "parens", "braces",
  "func_braces", "model_braces", "prog_braces", "raw", "double_braces",
  "$@1", "dim_reference", "var_reference", "func_reference",
  "dim_parameter", "var_parameter", "func_parameter", "var_member",
  "model_reference", "member_type", "parens_type", "any_type",
  "primary_type", "super_type", "list_type", "type", "dim_references",
  "dim_parameters", "reference_expression", "parens_expression",
  "this_expression", "primary_expression", "brackets_expression",
  "traversal_operator", "traversal_expression", "parameter_expression",
  "unary_operator", "unary_expression", "pow_operator", "pow_expression",
  "multiplicative_operator", "multiplicative_expression",
  "additive_operator", "additive_expression", "range_operator",
  "range_expression", "evaluate_operator", "evaluate_expression",
  "relational_operator", "relational_expression", "equality_operator",
  "equality_expression", "logical_and_operator", "logical_and_expression",
  "logical_or_operator", "logical_or_expression", "list_operator",
  "list_expression", "condition_operator", "condition_expression",
  "assignment_operator", "assignment_expression", "expression",
  "optional_expression", "option", "options", "optional_options",
  "parens_options", "var_declaration", "var_member_declaration",
  "func_declaration", "base", "model_declaration_param",
  "model_declaration", "prog_declaration_param", "prog_declaration",
  "expression_statement", "if", "while", "cpp", "hpp", "statement",
  "statements", "optional_statements", "func_statement", "func_statements",
  "optional_func_statements", "model_statement", "model_statements",
  "optional_model_statements", "prog_statement", "prog_statements",
  "optional_prog_statements", "import", "imports", "optional_imports",
  "first_file_statement", "file_statement", "file_statements",
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
     285,   286,   287,   288,    46,    40,    41,   123,   125,    58,
      59,    91,    93,    94,    44,    43,    45,    33,   126,    42,
      47,    37,    60,    62,   124,    61
};
# endif

#define YYPACT_NINF -242

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-242)))

#define YYTABLE_NINF -1

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     134,    79,     6,     6,  -242,  -242,   134,  -242,    60,    27,
    -242,  -242,  -242,  -242,    10,     8,  -242,  -242,  -242,  -242,
      46,    46,    14,  -242,     4,    29,    33,  -242,  -242,  -242,
    -242,    41,  -242,  -242,    79,  -242,    65,    53,  -242,    34,
    -242,  -242,  -242,  -242,  -242,  -242,    14,  -242,  -242,  -242,
    -242,   -17,  -242,  -242,  -242,  -242,  -242,  -242,  -242,  -242,
    -242,  -242,  -242,  -242,    74,  -242,    97,  -242,   144,  -242,
      89,   131,    88,  -242,   122,    26,    75,   142,    -3,   135,
      69,  -242,    99,  -242,     4,   137,  -242,  -242,  -242,  -242,
       4,  -242,  -242,  -242,  -242,  -242,  -242,  -242,    41,  -242,
    -242,  -242,   143,    46,    35,    46,   124,  -242,    46,    46,
    -242,   157,    14,  -242,    14,  -242,   155,  -242,  -242,    14,
    -242,  -242,  -242,    14,  -242,  -242,  -242,    14,    14,  -242,
      14,  -242,  -242,  -242,  -242,    14,  -242,  -242,    14,  -242,
      14,  -242,  -242,    14,    14,  -242,    14,  -242,  -242,    14,
     144,   144,   107,  -242,  -242,  -242,    13,  -242,   158,    14,
     160,  -242,  -242,  -242,   159,  -242,   152,   163,   107,  -242,
    -242,  -242,   161,   165,   164,   166,  -242,  -242,  -242,   124,
    -242,   169,  -242,    92,   168,  -242,  -242,   175,   162,   177,
    -242,  -242,    89,   131,   140,  -242,   122,    26,    75,   142,
    -242,  -242,  -242,   103,   120,   177,   177,   173,  -242,  -242,
    -242,  -242,  -242,  -242,   107,  -242,   178,     4,     4,  -242,
     176,    14,     4,    46,  -242,  -242,  -242,   107,   179,    46,
     -23,    46,  -242,  -242,  -242,  -242,  -242,  -242,  -242,  -242,
    -242,  -242,  -242,  -242,   182,   182,  -242,  -242,  -242,  -242,
    -242,  -242,   184,   167,  -242,  -242,  -242,  -242,    46,  -242,
     183,   190,   107,   218,  -242,  -242,   129,   191,    46,    46,
     107,  -242,   189,     0,  -242,  -242,  -242,   185,   188,   195,
    -242,  -242,  -242,  -242,    46,  -242,  -242,  -242
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
     188,     0,     0,     0,   185,   186,   184,   187,   203,     0,
       6,     5,     4,     3,     7,     0,    24,   154,   155,   183,
       0,     0,     0,     2,     0,     0,     0,   189,   190,   192,
     191,   202,   204,     1,     0,   182,     0,     0,   148,     0,
     145,    62,     9,    10,    11,    12,     0,    74,    75,    76,
      77,    27,    13,    14,    15,    16,    63,    59,    60,    72,
     136,    65,    64,    66,    67,    70,    73,    78,     0,    81,
      86,    90,    93,    96,   102,   106,   109,   112,   115,   118,
     122,   124,     0,    46,     0,    42,    47,    48,    49,    32,
       0,   134,   195,   196,   198,   197,   193,   194,   199,   201,
       8,    22,     0,   132,     0,     0,   176,   144,     0,     0,
     143,     0,   126,    28,     0,    69,     0,    79,    80,     0,
      83,    84,    85,     0,    92,    88,    89,     0,     0,    95,
       0,   100,   101,    98,    99,     0,   104,   105,     0,   108,
       0,   111,   114,     0,     0,   117,     0,   120,   121,     0,
       0,     0,   169,    38,    37,    50,    52,    54,     0,     0,
      31,   200,    25,    23,     0,   129,   131,     0,   181,   147,
     146,    29,    57,     0,     0,     0,   170,   171,   172,   173,
     175,     0,   137,     0,     0,    61,   125,     0,     0,    27,
      71,    82,    87,    91,    94,    97,   103,   107,   110,   113,
     116,   119,   123,     0,     0,     0,     0,     0,   156,   157,
     158,   159,   160,   165,   166,   168,     0,     0,     0,    45,
       0,     0,     0,     0,   133,   177,   178,   180,     0,     0,
       0,     0,   135,   174,    20,   140,   139,   141,    17,    68,
      34,    33,    36,    35,     0,     0,   149,   167,    19,    51,
      53,    41,     0,   128,   130,   179,    21,    58,     0,   142,
      44,    40,   164,   152,   153,    30,     0,     0,     0,     0,
     161,   163,     0,     0,   127,   138,    26,    55,     0,     0,
     162,    18,   150,   151,     0,    43,    39,    56
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -242,    -8,  -242,   198,  -242,  -242,  -242,  -242,   -33,   -18,
    -235,   -12,  -180,  -242,  -242,   231,  -242,  -242,  -242,  -242,
    -242,    11,  -242,  -242,  -242,  -242,  -242,  -242,   -20,  -242,
      17,  -242,  -241,     7,  -242,  -242,  -242,  -242,   121,  -242,
    -242,   -38,  -242,   119,  -242,   116,  -242,   113,  -242,   114,
    -242,   111,  -242,   108,  -242,   106,  -242,   105,  -242,   104,
    -242,  -242,    80,   109,  -242,    20,  -242,  -242,   -21,  -242,
      25,  -242,  -242,  -242,   241,  -242,    -2,  -107,  -242,   243,
    -242,   244,  -242,   -19,  -242,     9,     5,  -160,   -15,  -242,
    -242,    42,  -242,  -242,    78,  -242,    31,  -242,  -242,  -242,
     253,  -242,  -242,  -242,   170,  -242,  -242
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,    51,    14,    15,    52,    53,    54,    55,    56,   113,
     263,   154,   110,   170,   102,    17,    36,   277,    57,    58,
     172,    59,    60,   175,    86,   261,    87,    88,   155,   156,
     157,   158,   278,   173,    61,    62,    63,    64,    65,   116,
      66,    67,    68,    69,   119,    70,   123,    71,   127,    72,
     128,    73,   130,    74,   135,    75,   138,    76,   140,    77,
     143,    78,   144,    79,   146,    80,   149,    81,   207,   187,
     165,   166,   167,   104,    92,   176,    93,   183,    40,    94,
      38,    95,   208,   209,   210,   211,   212,   213,   271,   272,
     214,   215,   216,   179,   180,   181,   226,   227,   228,     6,
       7,     8,    31,    98,    99,    32,     9
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint16 yytable[] =
{
      25,    82,   184,   236,    89,     5,    28,   205,   225,     4,
     264,     5,    37,    39,   106,     4,    85,    83,   112,    26,
      16,    23,    90,    25,   141,   111,    41,    33,   279,   258,
     117,    23,    42,    43,    44,    45,    97,   262,   282,    84,
      96,   142,    26,   287,    34,    20,    21,    22,    35,    46,
     259,     2,     3,    24,   131,   132,   217,   218,    23,    47,
      48,    49,    50,    23,    20,    21,    22,   225,    90,   105,
     160,   106,   168,    91,   107,   169,    85,    23,   133,   134,
      24,   101,    85,    10,    11,    12,   108,   275,   103,   109,
      25,   186,   147,   188,   148,   164,    13,   171,   174,    24,
     182,   182,   270,    97,   177,   136,   137,    96,   189,    26,
     270,   178,   203,   204,   205,   114,   206,     2,     3,    41,
     124,   150,   118,   151,    23,    42,    43,    44,    45,   106,
      22,   115,   235,   125,   126,     3,   152,     1,   220,   153,
     152,    23,    46,   240,     2,     3,    24,    42,    43,    44,
      45,   267,    47,    48,    49,    50,    41,   152,   162,   163,
     242,    23,    42,    43,    44,    45,   201,    41,   139,   202,
     129,   174,    23,    42,    43,    44,    45,   177,   159,    46,
     120,   121,   122,    24,   178,   125,   126,   244,   245,   145,
      46,   241,   243,   185,   219,   221,   142,   249,   222,   224,
     252,   230,   253,   231,   239,   229,   232,   234,   237,    85,
      85,   238,   112,   246,    85,   164,   248,   256,   251,   262,
     265,   171,   266,   260,   268,   269,   273,   281,   106,   284,
     285,   286,   100,   274,    18,   250,   257,   190,   191,   192,
     193,   195,   194,   196,   197,   198,   223,   199,   254,    27,
     182,    29,    30,   200,   283,   280,   247,   233,   255,    19,
     276,   276,     0,     0,     0,     0,     0,     0,   161,     0,
       0,     0,     0,     0,     0,     0,   276
};

static const yytype_int16 yycheck[] =
{
       8,    22,   109,   183,    24,     0,     8,     7,   168,     0,
     245,     6,    20,    21,    37,     6,    24,    13,    35,     8,
      14,    17,    39,    31,    27,    46,    12,     0,   269,    52,
      68,    17,    18,    19,    20,    21,    31,    37,   273,    35,
      31,    44,    31,   284,    34,     4,     5,     6,    40,    35,
     230,    10,    11,    39,    28,    29,    43,    44,    17,    45,
      46,    47,    48,    17,     4,     5,     6,   227,    39,    35,
      90,    37,    37,    40,    40,    40,    84,    17,    52,    53,
      39,    16,    90,     4,     5,     6,    52,   267,    35,    55,
      98,   112,    23,   114,    25,   103,    17,   105,   106,    39,
     108,   109,   262,    98,   106,    30,    31,    98,   116,    98,
     270,   106,   150,   151,     7,    41,     9,    10,    11,    12,
      32,    22,    33,    24,    17,    18,    19,    20,    21,    37,
       6,    34,    40,    45,    46,    11,    37,     3,   159,    40,
      37,    17,    35,    40,    10,    11,    39,    18,    19,    20,
      21,   258,    45,    46,    47,    48,    12,    37,    15,    16,
      40,    17,    18,    19,    20,    21,   146,    12,    26,   149,
      48,   179,    17,    18,    19,    20,    21,   179,    41,    35,
      49,    50,    51,    39,   179,    45,    46,   205,   206,    54,
      35,   203,   204,    36,    36,    35,    44,   217,    39,    36,
     221,    36,   222,    39,    42,    44,    40,    38,    40,   217,
     218,    36,    35,    40,   222,   223,    38,    38,    42,    37,
      36,   229,    55,   231,    41,    35,     8,    38,    37,    44,
      42,    36,    34,   266,     3,   218,   229,   116,   119,   123,
     127,   130,   128,   135,   138,   140,   166,   143,   223,     8,
     258,     8,     8,   144,   273,   270,   214,   179,   227,     6,
     268,   269,    -1,    -1,    -1,    -1,    -1,    -1,    98,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   284
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,    10,    11,   141,   142,   155,   156,   157,   162,
       4,     5,     6,    17,    58,    59,    14,    71,    71,   156,
       4,     5,     6,    17,    39,    57,    77,   130,   132,   135,
     137,   158,   161,     0,    34,    40,    72,    57,   136,    57,
     134,    12,    18,    19,    20,    21,    35,    45,    46,    47,
      48,    57,    60,    61,    62,    63,    64,    74,    75,    77,
      78,    90,    91,    92,    93,    94,    96,    97,    98,    99,
     101,   103,   105,   107,   109,   111,   113,   115,   117,   119,
     121,   123,   124,    13,    35,    57,    80,    82,    83,    84,
      39,    40,   130,   132,   135,   137,   141,   142,   159,   160,
      59,    16,    70,    35,   129,    35,    37,    40,    52,    55,
      68,   124,    35,    65,    41,    34,    95,    97,    33,   100,
      49,    50,    51,   102,    32,    45,    46,   104,   106,    48,
     108,    28,    29,    52,    53,   110,    30,    31,   112,    26,
     114,    27,    44,   116,   118,    54,   120,    23,    25,   122,
      22,    24,    37,    40,    67,    84,    85,    86,    87,    41,
      84,   160,    15,    16,    57,   126,   127,   128,    37,    40,
      69,    57,    76,    89,    57,    79,   131,   132,   142,   149,
     150,   151,    57,   133,   133,    36,   124,   125,   124,    57,
      94,    99,   101,   103,   105,   107,   109,   111,   113,   115,
     119,   121,   121,    97,    97,     7,     9,   124,   138,   139,
     140,   141,   142,   143,   146,   147,   148,    43,    44,    36,
     124,    35,    39,   118,    36,   143,   152,   153,   154,    44,
      36,    39,    40,   150,    38,    40,    68,    40,    36,    42,
      40,    67,    40,    67,    65,    65,    40,   147,    38,    84,
      86,    42,   124,    84,   126,   152,    38,    89,    52,    68,
      57,    81,    37,    66,    66,    36,    55,   133,    41,    35,
     143,   144,   145,     8,    64,    68,    57,    73,    88,    88,
     144,    38,    66,   139,    44,    42,    36,    88
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    56,    57,    58,    58,    58,    58,    59,    59,    60,
      61,    62,    63,    64,    64,    64,    64,    65,    66,    67,
      68,    69,    70,    70,    72,    71,    73,    74,    75,    76,
      77,    77,    77,    78,    78,    78,    78,    78,    78,    79,
      79,    80,    80,    81,    81,    82,    83,    84,    84,    84,
      85,    85,    86,    86,    87,    88,    88,    89,    89,    90,
      90,    91,    92,    93,    93,    93,    93,    94,    94,    95,
      96,    96,    97,    97,    98,    98,    98,    98,    99,    99,
     100,   101,   101,   102,   102,   102,   103,   103,   104,   104,
     105,   105,   106,   107,   107,   108,   109,   109,   110,   110,
     110,   110,   111,   111,   112,   112,   113,   113,   114,   115,
     115,   116,   117,   117,   118,   119,   119,   120,   121,   121,
     122,   122,   123,   123,   124,   125,   125,   126,   126,   127,
     127,   128,   128,   129,   130,   131,   132,   133,   134,   134,
     134,   134,   134,   134,   134,   135,   136,   136,   137,   138,
     139,   139,   139,   140,   141,   142,   143,   143,   143,   143,
     143,   144,   144,   145,   145,   146,   147,   147,   148,   148,
     149,   149,   149,   150,   150,   151,   151,   152,   153,   153,
     154,   154,   155,   156,   156,   156,   156,   157,   157,   158,
     158,   158,   158,   159,   159,   159,   159,   159,   159,   160,
     160,   161,   161,   161,   162
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     1,     1,     1,     1,     1,     3,     1,
       1,     1,     1,     1,     1,     1,     1,     3,     3,     3,
       3,     3,     1,     2,     0,     4,     1,     1,     2,     1,
       6,     3,     2,     4,     4,     4,     4,     2,     2,     6,
       3,     4,     1,     4,     1,     3,     1,     1,     1,     1,
       1,     3,     1,     3,     1,     1,     3,     1,     3,     1,
       1,     3,     1,     1,     1,     1,     1,     1,     4,     1,
       1,     3,     1,     1,     1,     1,     1,     1,     1,     2,
       1,     1,     3,     1,     1,     1,     1,     3,     1,     1,
       1,     3,     1,     1,     3,     1,     1,     3,     1,     1,
       1,     1,     1,     3,     1,     1,     1,     3,     1,     1,
       3,     1,     1,     3,     1,     1,     3,     1,     1,     3,
       1,     1,     1,     3,     1,     1,     0,     5,     3,     1,
       3,     1,     0,     3,     2,     2,     2,     1,     7,     4,
       4,     4,     5,     2,     2,     2,     3,     3,     2,     2,
       5,     5,     3,     3,     2,     2,     1,     1,     1,     1,
       1,     1,     2,     1,     0,     1,     1,     2,     1,     0,
       1,     1,     1,     1,     2,     1,     0,     1,     1,     2,
       1,     0,     3,     2,     1,     1,     1,     1,     0,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       2,     2,     1,     0,     2
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
#line 137 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name((yyvsp[0].valString), make_loc((yyloc))); }
#line 1664 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 3:
#line 141 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name((yyvsp[0].valString), make_loc((yyloc))); }
#line 1670 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 4:
#line 142 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("function", make_loc((yyloc))); }
#line 1676 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 5:
#line 143 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("model", make_loc((yyloc))); }
#line 1682 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 6:
#line 144 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("program", make_loc((yyloc))); }
#line 1688 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 7:
#line 148 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valPath) = new bi::Path((yyvsp[0].valName), nullptr, make_loc((yyloc))); }
#line 1694 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 8:
#line 149 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valPath) = new bi::Path((yyvsp[-2].valName), (yyvsp[0].valPath), make_loc((yyloc))); }
#line 1700 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 9:
#line 158 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BooleanLiteral((yyvsp[0].valBool), yytext, new bi::ModelReference(new bi::Name("Boolean")), make_loc((yyloc))); }
#line 1706 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 10:
#line 162 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::IntegerLiteral((yyvsp[0].valInt), yytext, new bi::ModelReference(new bi::Name("Integer")), make_loc((yyloc))); }
#line 1712 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 11:
#line 166 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::RealLiteral((yyvsp[0].valReal), yytext, new bi::ModelReference(new bi::Name("Real")), make_loc((yyloc))); }
#line 1718 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 12:
#line 170 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::StringLiteral((yyvsp[0].valString), yytext, new bi::ModelReference(new bi::Name("String")), make_loc((yyloc))); }
#line 1724 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 17:
#line 186 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ParenthesesExpression((yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 1730 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 18:
#line 190 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracesExpression((yyvsp[-1].valStatement), make_loc((yyloc))); }
#line 1736 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 19:
#line 194 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracesExpression((yyvsp[-1].valStatement), make_loc((yyloc))); }
#line 1742 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 20:
#line 198 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracesExpression((yyvsp[-1].valStatement), make_loc((yyloc))); }
#line 1748 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 21:
#line 202 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracesExpression((yyvsp[-1].valStatement), make_loc((yyloc))); }
#line 1754 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 22:
#line 211 "parser.ypp" /* yacc.c:1661  */
    { raw << (yyvsp[0].valString); }
#line 1760 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 23:
#line 212 "parser.ypp" /* yacc.c:1661  */
    { raw << (yyvsp[0].valString); }
#line 1766 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 24:
#line 216 "parser.ypp" /* yacc.c:1661  */
    { raw.str(""); }
#line 1772 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 26:
#line 225 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::DimReference((yyvsp[0].valName), make_loc((yyloc))); }
#line 1778 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 27:
#line 229 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::VarReference((yyvsp[0].valName), make_loc((yyloc))); }
#line 1784 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 28:
#line 233 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::FuncReference((yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1790 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 29:
#line 242 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::DimParameter((yyvsp[0].valName), make_loc((yyloc))); }
#line 1796 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 30:
#line 246 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::VarParameter((yyvsp[-5].valName), (yyvsp[-3].valType), (yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 1802 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 31:
#line 247 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::VarParameter((yyvsp[-2].valName), (yyvsp[0].valType), new bi::EmptyExpression(), make_loc((yyloc))); }
#line 1808 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 32:
#line 248 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::VarParameter(new bi::Name(), (yyvsp[0].valType), new bi::EmptyExpression(), make_loc((yyloc))); }
#line 1814 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 33:
#line 252 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = convertReference((yyvsp[-3].valExpression), (yyvsp[-1].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1820 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 34:
#line 253 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = convertReference((yyvsp[-3].valExpression), (yyvsp[-1].valExpression), new bi::EmptyExpression(), make_loc((yyloc))); }
#line 1826 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 35:
#line 254 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = convertReference((yyvsp[-3].valExpression), (yyvsp[-1].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1832 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 36:
#line 255 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = convertReference((yyvsp[-3].valExpression), (yyvsp[-1].valExpression), new bi::EmptyExpression(), make_loc((yyloc))); }
#line 1838 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 37:
#line 256 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = convertReference((yyvsp[-1].valExpression), new bi::EmptyExpression(), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1844 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 38:
#line 257 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = convertReference((yyvsp[-1].valExpression), new bi::EmptyExpression(), new bi::EmptyExpression(), make_loc((yyloc))); }
#line 1850 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 39:
#line 261 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::VarMember((yyvsp[-5].valName), (yyvsp[-3].valType), (yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 1856 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 40:
#line 262 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::VarMember((yyvsp[-2].valName), (yyvsp[0].valType), new bi::EmptyExpression(), make_loc((yyloc))); }
#line 1862 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 41:
#line 271 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelReference((yyvsp[-3].valName), (yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 1868 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 42:
#line 272 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelReference((yyvsp[0].valName), new bi::EmptyExpression(), make_loc((yyloc))); }
#line 1874 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 43:
#line 276 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelReference((yyvsp[-3].valName), (yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 1880 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 44:
#line 277 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelReference((yyvsp[0].valName), new bi::EmptyExpression(), make_loc((yyloc))); }
#line 1886 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 45:
#line 281 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ParenthesesType((yyvsp[-1].valType), make_loc((yyloc))); }
#line 1892 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 46:
#line 285 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::EmptyType(); }
#line 1898 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 51:
#line 296 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::SuperType((yyvsp[-2].valType), (yyvsp[0].valType), make_loc((yyloc))); }
#line 1904 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 53:
#line 301 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::TypeList((yyvsp[-2].valType), (yyvsp[0].valType), make_loc((yyloc))); }
#line 1910 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 56:
#line 315 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ExpressionList((yyvsp[-2].valExpression), bi::OP_COMMA, (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1916 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 58:
#line 320 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ExpressionList((yyvsp[-2].valExpression), bi::OP_COMMA, (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1922 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 61:
#line 334 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ParenthesesExpression((yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 1928 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 62:
#line 338 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::This(make_loc((yyloc))); }
#line 1934 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 68:
#line 350 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracketsExpression((yyvsp[-3].valExpression), (yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 1940 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 69:
#line 354 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_TRAVERSE; }
#line 1946 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 71:
#line 359 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::Traversal((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1952 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 74:
#line 368 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_POS; }
#line 1958 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 75:
#line 369 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_NEG; }
#line 1964 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 76:
#line 370 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_NOT; }
#line 1970 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 77:
#line 371 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_TILDE; }
#line 1976 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 79:
#line 376 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::UnaryReference((yyvsp[-1].valOperator), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1982 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 80:
#line 380 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_POW; }
#line 1988 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 82:
#line 385 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BinaryReference((yyvsp[-2].valExpression), (yyvsp[-1].valOperator), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1994 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 83:
#line 389 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_MUL; }
#line 2000 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 84:
#line 390 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_DIV; }
#line 2006 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 85:
#line 391 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_MOD; }
#line 2012 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 87:
#line 396 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BinaryReference((yyvsp[-2].valExpression), (yyvsp[-1].valOperator), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2018 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 88:
#line 400 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_ADD; }
#line 2024 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 89:
#line 401 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_SUB; }
#line 2030 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 91:
#line 406 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BinaryReference((yyvsp[-2].valExpression), (yyvsp[-1].valOperator), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2036 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 92:
#line 410 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_RANGE; }
#line 2042 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 94:
#line 415 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::Range((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2048 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 95:
#line 419 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_TILDE; }
#line 2054 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 97:
#line 424 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BinaryReference((yyvsp[-2].valExpression), (yyvsp[-1].valOperator), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2060 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 98:
#line 428 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_LT; }
#line 2066 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 99:
#line 429 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_GT; }
#line 2072 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 100:
#line 430 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_LE; }
#line 2078 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 101:
#line 431 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_GE; }
#line 2084 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 103:
#line 436 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BinaryReference((yyvsp[-2].valExpression), (yyvsp[-1].valOperator), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2090 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 104:
#line 440 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_EQ; }
#line 2096 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 105:
#line 441 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_NE; }
#line 2102 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 107:
#line 446 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BinaryReference((yyvsp[-2].valExpression), (yyvsp[-1].valOperator), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2108 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 108:
#line 450 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_AND; }
#line 2114 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 110:
#line 455 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BinaryReference((yyvsp[-2].valExpression), (yyvsp[-1].valOperator), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2120 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 111:
#line 459 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_OR; }
#line 2126 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 113:
#line 464 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BinaryReference((yyvsp[-2].valExpression), (yyvsp[-1].valOperator), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2132 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 114:
#line 468 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_COMMA; }
#line 2138 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 116:
#line 473 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ExpressionList((yyvsp[-2].valExpression), (yyvsp[-1].valOperator), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2144 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 117:
#line 477 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_CONDITION; }
#line 2150 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 119:
#line 482 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ExpressionList((yyvsp[-2].valExpression), (yyvsp[-1].valOperator), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2156 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 120:
#line 486 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_LEFT_ARROW; }
#line 2162 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 121:
#line 487 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valOperator) = bi::OP_LEFT_TILDE_ARROW; }
#line 2168 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 123:
#line 492 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BinaryReference((yyvsp[-2].valExpression), (yyvsp[-1].valOperator), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2174 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 126:
#line 501 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::EmptyExpression(); }
#line 2180 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 127:
#line 510 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::VarParameter((yyvsp[-4].valName), (yyvsp[-2].valType), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2186 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 128:
#line 511 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::VarParameter((yyvsp[-2].valName), (yyvsp[0].valType), new bi::EmptyExpression(), make_loc((yyloc))); }
#line 2192 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 130:
#line 516 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ExpressionList((yyvsp[-2].valExpression), (yyvsp[-1].valOperator), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2198 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 132:
#line 521 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::EmptyExpression(); }
#line 2204 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 133:
#line 525 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ParenthesesExpression((yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 2210 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 134:
#line 534 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::VarDeclaration(dynamic_cast<bi::VarParameter*>((yyvsp[-1].valExpression)), make_loc((yyloc))); }
#line 2216 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 135:
#line 538 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::VarDeclaration(dynamic_cast<bi::VarParameter*>((yyvsp[-1].valExpression)), make_loc((yyloc))); }
#line 2222 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 136:
#line 542 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::FuncDeclaration(dynamic_cast<bi::OverloadableParameter*>((yyvsp[0].valExpression)), make_loc((yyloc))); }
#line 2228 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 137:
#line 546 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelReference((yyvsp[0].valName), make_loc((yyloc))); }
#line 2234 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 138:
#line 550 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-6].valName), (yyvsp[-4].valExpression), bi::OP_LT, (yyvsp[-1].valType), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2240 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 139:
#line 551 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-3].valName), new bi::EmptyExpression(), bi::OP_LT, (yyvsp[-1].valType), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2246 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 140:
#line 552 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-3].valName), new bi::EmptyExpression(), bi::OP_LT, (yyvsp[-1].valType), new bi::EmptyExpression(), make_loc((yyloc))); }
#line 2252 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 141:
#line 553 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-3].valName), new bi::EmptyExpression(), bi::OP_EQUALS, (yyvsp[-1].valType), new bi::EmptyExpression(), make_loc((yyloc))); }
#line 2258 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 142:
#line 554 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-4].valName), (yyvsp[-2].valExpression), bi::OP_EMPTY, new bi::EmptyType(), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2264 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 143:
#line 555 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-1].valName), new bi::EmptyExpression(), bi::OP_EMPTY, new bi::EmptyType(), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2270 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 144:
#line 556 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-1].valName), new bi::EmptyExpression(), bi::OP_EMPTY, new bi::EmptyType(), new bi::EmptyExpression(), make_loc((yyloc))); }
#line 2276 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 145:
#line 560 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::ModelDeclaration(dynamic_cast<bi::ModelParameter*>((yyvsp[0].valType)), make_loc((yyloc))); }
#line 2282 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 146:
#line 564 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valProg) = new bi::ProgParameter((yyvsp[-2].valName), (yyvsp[-1].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2288 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 147:
#line 565 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valProg) = new bi::ProgParameter((yyvsp[-2].valName), (yyvsp[-1].valExpression), new bi::EmptyExpression(), make_loc((yyloc))); }
#line 2294 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 148:
#line 569 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::ProgDeclaration(dynamic_cast<bi::ProgParameter*>((yyvsp[0].valProg)), make_loc((yyloc))); }
#line 2300 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 149:
#line 573 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::ExpressionStatement((yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 2306 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 150:
#line 577 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Conditional((yyvsp[-3].valExpression), (yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2312 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 151:
#line 578 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Conditional((yyvsp[-3].valExpression), (yyvsp[-2].valExpression), new bi::BracesExpression((yyvsp[0].valStatement)), make_loc((yyloc))); }
#line 2318 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 152:
#line 579 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Conditional((yyvsp[-1].valExpression), (yyvsp[0].valExpression), new bi::EmptyExpression(), make_loc((yyloc))); }
#line 2324 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 153:
#line 583 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Loop((yyvsp[-1].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2330 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 154:
#line 587 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Raw(new bi::Name("cpp"), raw.str(), make_loc((yyloc))); }
#line 2336 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 155:
#line 591 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Raw(new bi::Name("hpp"), raw.str(), make_loc((yyloc))); }
#line 2342 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 162:
#line 604 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), bi::OP_SEMICOLON, (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2348 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 164:
#line 609 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2354 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 167:
#line 618 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), bi::OP_SEMICOLON, (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2360 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 169:
#line 623 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2366 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 174:
#line 634 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), bi::OP_SEMICOLON, (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2372 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 176:
#line 639 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2378 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 179:
#line 648 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), bi::OP_SEMICOLON, (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2384 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 181:
#line 653 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2390 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 182:
#line 657 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Import((yyvsp[-1].valPath), compiler->import((yyvsp[-1].valPath)), make_loc((yyloc))); }
#line 2396 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 183:
#line 661 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), bi::OP_SEMICOLON, (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2402 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 188:
#line 669 "parser.ypp" /* yacc.c:1661  */
    {(yyval.valStatement) = new bi::EmptyStatement(); }
#line 2408 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 200:
#line 690 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), bi::OP_SEMICOLON, (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2414 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 201:
#line 694 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), bi::OP_SEMICOLON, (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2420 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 203:
#line 696 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2426 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 204:
#line 700 "parser.ypp" /* yacc.c:1661  */
    { compiler->setRoot((yyvsp[-1].valStatement), (yyvsp[0].valStatement)); }
#line 2432 "parser.tab.cpp" /* yacc.c:1661  */
    break;


#line 2436 "parser.tab.cpp" /* yacc.c:1661  */
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
#line 703 "parser.ypp" /* yacc.c:1906  */

