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
    RESULT_OP = 277,
    FORWARD_OP = 278,
    BACKWARD_OP = 279,
    SIMULATE_OP = 280,
    CONDITION_OP = 281,
    AND_OP = 282,
    OR_OP = 283,
    LE_OP = 284,
    GE_OP = 285,
    EQ_OP = 286,
    NE_OP = 287,
    RANGE_OP = 288
  };
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 72 "parser.ypp" /* yacc.c:355  */

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

#line 162 "parser.tab.cpp" /* yacc.c:355  */
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

#line 193 "parser.tab.cpp" /* yacc.c:358  */
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

  bi::Expression* make_binary(bi::Expression* left, bi::shared_ptr<bi::Name> op, bi::Expression* right, bi::Location* loc = nullptr) {
    return new bi::FuncReference(op, new bi::ParenthesesExpression(new bi::ExpressionList(left, right)), bi::BINARY_OPERATOR, loc);
  }

  bi::Expression* make_unary(bi::shared_ptr<bi::Name> op, bi::Expression* right, bi::Location* loc = nullptr) {
    return new bi::FuncReference(op, new bi::ParenthesesExpression(right), bi::UNARY_OPERATOR, loc);
  }

  bi::Expression* make_assign(bi::Expression* left, bi::shared_ptr<bi::Name> op, bi::Expression* right, bi::Location* loc = nullptr) {
    return new bi::FuncReference(op, new bi::ParenthesesExpression(new bi::ExpressionList(left, right)), bi::ASSIGNMENT_OPERATOR, loc);
  }

  bi::Expression* convert_ref(bi::Expression* ref,
    bi::Expression* result, bi::Expression* braces, bi::Location* loc) {
    bi::FuncReference* func = dynamic_cast<bi::FuncReference*>(ref);
    assert(func);
    bi::Expression* param;
    
    if (func) {
      param = new bi::FuncParameter(func->name, func->parens.release(), result, braces, func->form, loc);
    } else {
      assert(false);
    }
    delete ref;
    
    return param;
  }
  
  bi::VarParameter* init_param(bi::Expression* expr, bi::Expression* value) {
    bi::VarParameter* var = dynamic_cast<bi::VarParameter*>(expr);
    assert(var);
    var->value = value;
    return var;
  }

#line 258 "parser.tab.cpp" /* yacc.c:359  */

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
#define YYFINAL  86
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   229

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  54
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  99
/* YYNRULES -- Number of rules.  */
#define YYNRULES  183
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  253

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
       2,     2,     2,    46,     2,     2,     2,     2,     2,     2,
      35,    36,    50,    48,    53,    49,    34,    51,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    41,    42,
      43,    44,    52,     2,    45,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    37,     2,    38,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    39,     2,    40,    47,     2,     2,     2,
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
       0,   149,   149,   153,   154,   155,   156,   160,   161,   170,
     174,   178,   182,   186,   187,   188,   189,   198,   202,   203,
     207,   211,   215,   219,   223,   232,   233,   237,   237,   246,
     247,   251,   252,   253,   254,   258,   262,   266,   270,   271,
     272,   273,   274,   275,   276,   280,   281,   290,   294,   298,
     307,   311,   312,   316,   317,   321,   322,   326,   327,   331,
     332,   336,   341,   350,   351,   355,   359,   363,   364,   365,
     366,   370,   371,   375,   376,   380,   381,   385,   389,   390,
     394,   395,   399,   400,   401,   405,   406,   410,   411,   415,
     416,   420,   421,   425,   426,   430,   431,   432,   433,   437,
     438,   442,   443,   447,   448,   452,   456,   457,   461,   465,
     466,   470,   471,   475,   476,   480,   481,   485,   486,   487,
     491,   495,   496,   500,   504,   505,   514,   515,   519,   520,
     524,   525,   529,   538,   539,   543,   547,   551,   555,   559,
     560,   561,   565,   569,   573,   577,   578,   579,   580,   581,
     585,   586,   590,   591,   595,   599,   600,   604,   605,   609,
     610,   611,   615,   616,   620,   621,   625,   629,   630,   634,
     635,   639,   643,   644,   645,   646,   647,   648,   649,   653,
     654,   658,   659,   663
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
  "REAL_LITERAL", "STRING_LITERAL", "RESULT_OP", "FORWARD_OP",
  "BACKWARD_OP", "SIMULATE_OP", "CONDITION_OP", "AND_OP", "OR_OP", "LE_OP",
  "GE_OP", "EQ_OP", "NE_OP", "RANGE_OP", "'.'", "'('", "')'", "'['", "']'",
  "'{'", "'}'", "':'", "';'", "'<'", "'='", "'@'", "'!'", "'~'", "'+'",
  "'-'", "'*'", "'/'", "'>'", "','", "$accept", "name", "path_name",
  "path", "bool_literal", "int_literal", "real_literal", "string_literal",
  "literal", "parens", "optional_parens", "brackets", "braces",
  "func_braces", "model_braces", "prog_braces", "raw", "double_braces",
  "$@1", "var_parameter", "func_parameter", "base", "base_less_operator",
  "base_equal_operator", "model_parameter", "prog_parameter",
  "var_reference", "func_reference", "model_reference", "parens_type",
  "primary_type", "brackets_type", "lambda_type", "assignable_type",
  "random_type", "list_type", "type", "reference_expression",
  "parens_expression", "this_expression", "primary_expression",
  "index_expression", "index_list", "brackets_expression",
  "member_operator", "member_expression", "parameter_expression",
  "unary_operator", "unary_expression", "multiplicative_operator",
  "multiplicative_expression", "additive_operator", "additive_expression",
  "relational_operator", "relational_expression", "equality_operator",
  "equality_expression", "logical_and_operator", "logical_and_expression",
  "logical_or_operator", "logical_or_expression", "backward_operator",
  "backward_expression", "forward_operator", "forward_expression",
  "list_operator", "list_expression", "expression", "optional_expression",
  "option", "options", "optional_options", "parens_options",
  "var_declaration", "func_declaration", "model_declaration",
  "prog_declaration", "expression_statement", "if", "while", "cpp", "hpp",
  "statement", "statements", "optional_statements", "func_statement",
  "func_statements", "optional_func_statements", "model_statement",
  "model_statements", "optional_model_statements", "prog_statement",
  "prog_statements", "optional_prog_statements", "import",
  "file_statement", "file_statements", "optional_file_statements", "file", YY_NULLPTR
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
     285,   286,   287,   288,    46,    40,    41,    91,    93,   123,
     125,    58,    59,    60,    61,    64,    33,   126,    43,    45,
      42,    47,    62,    44
};
# endif

#define YYPACT_NINF -202

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-202)))

#define YYTABLE_NINF -1

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     105,   108,    -3,    -3,   109,     8,     8,  -202,     3,   -17,
      38,  -202,  -202,  -202,  -202,  -202,  -202,  -202,   105,  -202,
    -202,    75,  -202,  -202,  -202,  -202,    52,    49,    72,  -202,
     144,  -202,  -202,  -202,  -202,  -202,  -202,   109,  -202,  -202,
    -202,    60,  -202,  -202,  -202,  -202,  -202,  -202,  -202,  -202,
    -202,  -202,  -202,  -202,    82,  -202,    97,  -202,   121,  -202,
      67,    16,    15,   120,   107,   156,    12,    83,  -202,    27,
    -202,  -202,  -202,     3,    59,   102,  -202,  -202,   106,  -202,
     113,  -202,     3,   109,  -202,  -202,  -202,   108,  -202,    10,
      45,   109,    19,  -202,  -202,  -202,    11,  -202,    -3,    -3,
     112,  -202,   109,  -202,   157,  -202,  -202,  -202,   109,  -202,
    -202,   109,  -202,  -202,  -202,  -202,   109,  -202,  -202,   109,
    -202,   109,  -202,  -202,  -202,   109,   109,  -202,  -202,   109,
     109,  -202,   109,   121,    22,  -202,  -202,   137,   123,  -202,
    -202,   135,  -202,  -202,  -202,   109,  -202,  -202,  -202,   130,
    -202,   150,  -202,    83,   149,    22,  -202,  -202,  -202,   153,
    -202,  -202,  -202,    19,  -202,   141,  -202,    -3,   102,    63,
     148,  -202,    83,   155,   158,   102,  -202,  -202,    67,    16,
      15,   120,   107,  -202,  -202,  -202,  -202,    81,   102,   102,
     152,  -202,  -202,  -202,  -202,  -202,  -202,    22,  -202,   159,
    -202,   145,     3,  -202,   160,  -202,   109,    10,  -202,  -202,
    -202,    22,   161,  -202,  -202,  -202,   164,  -202,  -202,  -202,
    -202,   109,  -202,   109,  -202,  -202,   165,   165,  -202,  -202,
    -202,  -202,  -202,  -202,  -202,  -202,  -202,  -202,  -202,  -202,
    -202,  -202,    22,   187,  -202,    22,  -202,   166,    14,  -202,
    -202,  -202,  -202
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
     182,     0,     0,     0,     0,     0,     0,     2,     0,     0,
       0,   175,   176,   178,   177,   173,   174,   172,   179,   181,
     183,     0,     6,     5,     4,     3,     7,     0,     0,   137,
       0,   136,    66,     9,    10,    11,    12,     0,    84,    82,
      83,    47,    13,    14,    15,    16,    67,    80,   135,    63,
      64,    69,    68,    70,    75,    78,    81,    85,     0,    89,
      93,    99,   103,   106,   109,   113,   117,   121,   123,     0,
      27,   143,   144,     0,     0,    19,    51,    52,    53,    55,
      57,    30,     0,     0,   134,   180,     1,     0,   171,   131,
       0,   125,   165,    44,    36,    37,     0,    43,     0,     0,
       0,    48,     0,    77,     0,    86,    87,    88,     0,    91,
      92,     0,    97,    98,    95,    96,     0,   101,   102,     0,
     105,     0,   111,   112,   108,     0,     0,   115,   116,     0,
       0,   120,     0,     0,   158,    34,    33,     0,    59,    61,
      62,     0,    56,    18,    49,     0,    54,    58,    29,     0,
       8,   127,   128,   130,     0,   170,    46,    45,   124,     0,
     159,   160,   161,   162,   164,     0,    42,     0,    19,     0,
       0,    65,    73,     0,    72,    47,    79,    90,    94,   100,
     104,   107,   110,   114,   119,   118,   122,     0,     0,     0,
       0,   145,   146,   147,   148,   149,   154,   155,   157,     0,
      25,     0,     0,    50,     0,   133,     0,     0,   132,   166,
     167,   169,     0,    17,   163,    23,     0,    35,    40,    39,
      41,     0,    76,     0,    32,    31,     0,     0,   138,   156,
      22,    28,    26,    60,    20,   126,   129,   168,    24,    38,
      74,    71,   153,   141,   142,   150,   152,     0,     0,   151,
      21,   139,   140
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -202,     0,  -202,   110,  -202,  -202,  -202,  -202,  -202,   -23,
      28,  -202,  -201,    13,   -92,  -202,  -202,   199,  -202,     1,
    -202,   -88,   114,  -202,  -202,  -202,  -202,  -202,  -202,  -202,
    -202,   138,  -202,   -67,  -202,  -202,  -202,  -202,  -202,  -202,
    -202,  -202,   -12,   111,  -202,  -202,   -48,  -202,   103,  -202,
     115,  -202,    98,  -202,    94,  -202,    95,  -202,    92,  -202,
    -202,  -202,    93,  -202,   -74,   -91,    86,     9,  -202,    17,
    -202,  -202,  -202,   -80,   -75,  -202,  -202,  -202,   -28,  -202,
      78,     5,  -139,   -24,  -202,  -202,    25,  -202,  -202,    62,
    -202,    18,  -202,  -202,  -202,  -202,   205,  -202,  -202
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,    41,    26,    27,    42,    43,    44,    45,    46,   101,
     144,   146,   243,   136,    97,   157,   201,    71,   137,    47,
      48,   169,    98,    99,    31,    29,    49,    50,    76,    77,
      78,    79,    80,    81,   139,   140,   141,    51,    52,    53,
      54,   172,   173,    55,   104,    56,    57,    58,    59,   108,
      60,   111,    61,   116,    62,   119,    63,   121,    64,   125,
      65,   126,    66,   130,    67,   132,    68,   190,   159,   152,
     153,   154,    90,    11,    12,    13,    14,   191,   192,   193,
     194,   195,   196,   246,   247,   197,   198,   199,   163,   164,
     165,   210,   211,   212,    17,    18,    19,    20,    21
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint8 yytable[] =
{
       9,    10,    28,    30,   166,    16,   138,    96,    75,   149,
     105,   170,   160,    69,     7,   148,   209,   161,     9,    10,
       7,   188,    70,    16,    82,     4,   244,     7,   174,   188,
       6,   189,     5,     6,    32,   127,     7,   128,    73,     7,
      33,    34,    35,    36,   112,   113,   100,   251,    74,   133,
      92,     8,   143,   242,    94,   184,   185,    37,   114,   129,
       8,    83,   207,     8,   109,   110,   134,   115,    38,   135,
      39,    40,   209,    75,    75,    86,     7,   219,    15,   216,
      84,   221,    75,   160,   155,   187,    87,   156,   161,     9,
     151,    88,     9,    10,    73,    91,    15,   162,   168,   168,
     158,    82,    92,   245,   175,   218,   245,    89,     1,     2,
       3,     4,    22,    23,    24,     5,     6,   106,   107,   102,
     134,    32,     7,   224,   239,    25,     7,    33,    34,    35,
      36,   103,   235,    32,   120,   233,   131,    91,     7,    33,
      34,    35,    36,   145,    37,   143,     8,   174,   171,   241,
       8,   117,   118,   200,   204,    38,    37,    39,    40,   147,
     231,   232,     8,     9,    10,   226,   227,   168,   162,    32,
     202,   203,   205,   206,     7,    33,    34,    35,    36,    91,
     122,   215,   123,    92,   124,   208,    93,    94,    95,   213,
     220,   223,    37,   222,   228,   248,   217,   150,   234,   230,
     225,   238,    75,    92,   242,    72,   250,     9,   151,   240,
     167,   177,   142,   180,   179,   176,   181,   182,   186,   183,
     252,   249,   229,    85,   236,   214,   178,     0,     0,   237
};

static const yytype_int16 yycheck[] =
{
       0,     0,     2,     3,    96,     0,    73,    30,     8,    83,
      58,    99,    92,     4,    17,    82,   155,    92,    18,    18,
      17,     7,    14,    18,    41,     6,   227,    17,   102,     7,
      11,     9,    10,    11,    12,    23,    17,    25,    35,    17,
      18,    19,    20,    21,    29,    30,    37,   248,    45,    22,
      39,    41,    75,    39,    43,   129,   130,    35,    43,    47,
      41,    23,   153,    41,    48,    49,    39,    52,    46,    42,
      48,    49,   211,    73,    74,     0,    17,   169,     0,   167,
      42,   172,    82,   163,    39,   133,    34,    42,   163,    89,
      89,    42,    92,    92,    35,    35,    18,    92,    98,    99,
      91,    41,    39,   242,   104,    42,   245,    35,     3,     4,
       5,     6,     4,     5,     6,    10,    11,    50,    51,    37,
      39,    12,    17,    42,   216,    17,    17,    18,    19,    20,
      21,    34,   206,    12,    27,   202,    53,    35,    17,    18,
      19,    20,    21,    37,    35,   168,    41,   221,    36,   223,
      41,    31,    32,    16,   145,    46,    35,    48,    49,    46,
      15,    16,    41,   163,   163,   188,   189,   167,   163,    12,
      47,    36,    42,    23,    17,    18,    19,    20,    21,    35,
      24,    40,    26,    39,    28,    36,    42,    43,    44,    36,
      42,    33,    35,    38,    42,     8,   168,    87,    38,    40,
     187,    40,   202,    39,    39,     6,    40,   207,   207,   221,
      96,   108,    74,   119,   116,   104,   121,   125,   132,   126,
     248,   245,   197,    18,   207,   163,   111,    -1,    -1,   211
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,     5,     6,    10,    11,    17,    41,    55,
      73,   127,   128,   129,   130,   134,   135,   148,   149,   150,
     151,   152,     4,     5,     6,    17,    56,    57,    55,    79,
      55,    78,    12,    18,    19,    20,    21,    35,    46,    48,
      49,    55,    58,    59,    60,    61,    62,    73,    74,    80,
      81,    91,    92,    93,    94,    97,    99,   100,   101,   102,
     104,   106,   108,   110,   112,   114,   116,   118,   120,   121,
      14,    71,    71,    35,    45,    55,    82,    83,    84,    85,
      86,    87,    41,    23,    42,   150,     0,    34,    42,    35,
     126,    35,    39,    42,    43,    44,    63,    68,    76,    77,
     121,    63,    37,    34,    98,   100,    50,    51,   103,    48,
      49,   105,    29,    30,    43,    52,   107,    31,    32,   109,
      27,   111,    24,    26,    28,   113,   115,    23,    25,    47,
     117,    53,   119,    22,    39,    42,    67,    72,    87,    88,
      89,    90,    85,    63,    64,    37,    65,    46,    87,   118,
      57,    73,   123,   124,   125,    39,    42,    69,   121,   122,
     127,   128,   135,   142,   143,   144,    68,    76,    55,    75,
      75,    36,    95,    96,   118,    55,    97,   102,   104,   106,
     108,   110,   112,   116,   118,   118,   120,   100,     7,     9,
     121,   131,   132,   133,   134,   135,   136,   139,   140,   141,
      16,    70,    47,    36,   121,    42,    23,   119,    36,   136,
     145,   146,   147,    36,   143,    40,    75,    64,    42,    68,
      42,   119,    38,    33,    42,    67,    63,    63,    42,   140,
      40,    15,    16,    87,    38,   118,   123,   145,    40,    68,
      96,   118,    39,    66,    66,   136,   137,   138,     8,   137,
      40,    66,   132
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    54,    55,    56,    56,    56,    56,    57,    57,    58,
      59,    60,    61,    62,    62,    62,    62,    63,    64,    64,
      65,    66,    67,    68,    69,    70,    70,    72,    71,    73,
      73,    74,    74,    74,    74,    75,    76,    77,    78,    78,
      78,    78,    78,    78,    78,    79,    79,    80,    81,    82,
      83,    84,    84,    85,    85,    86,    86,    87,    87,    88,
      88,    89,    90,    91,    91,    92,    93,    94,    94,    94,
      94,    95,    95,    96,    96,    97,    97,    98,    99,    99,
     100,   100,   101,   101,   101,   102,   102,   103,   103,   104,
     104,   105,   105,   106,   106,   107,   107,   107,   107,   108,
     108,   109,   109,   110,   110,   111,   112,   112,   113,   114,
     114,   115,   115,   116,   116,   117,   117,   118,   118,   118,
     119,   120,   120,   121,   122,   122,   123,   123,   124,   124,
     125,   125,   126,   127,   127,   128,   129,   130,   131,   132,
     132,   132,   133,   134,   135,   136,   136,   136,   136,   136,
     137,   137,   138,   138,   139,   140,   140,   141,   141,   142,
     142,   142,   143,   143,   144,   144,   145,   146,   146,   147,
     147,   148,   149,   149,   149,   149,   149,   149,   149,   150,
     150,   151,   151,   152
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     1,     1,     1,     1,     1,     3,     1,
       1,     1,     1,     1,     1,     1,     1,     3,     1,     0,
       3,     3,     3,     3,     3,     1,     2,     0,     4,     3,
       2,     4,     4,     2,     2,     2,     1,     1,     5,     4,
       4,     4,     3,     2,     2,     3,     3,     1,     2,     2,
       3,     1,     1,     1,     2,     1,     2,     1,     2,     1,
       3,     1,     1,     1,     1,     3,     1,     1,     1,     1,
       1,     3,     1,     1,     3,     1,     4,     1,     1,     3,
       1,     1,     1,     1,     1,     1,     2,     1,     1,     1,
       3,     1,     1,     1,     3,     1,     1,     1,     1,     1,
       3,     1,     1,     1,     3,     1,     1,     3,     1,     1,
       3,     1,     1,     1,     3,     1,     1,     1,     3,     3,
       1,     1,     3,     1,     1,     0,     3,     1,     1,     3,
       1,     0,     3,     4,     2,     2,     2,     2,     2,     5,
       5,     3,     3,     2,     2,     1,     1,     1,     1,     1,
       1,     2,     1,     0,     1,     1,     2,     1,     0,     1,
       1,     1,     1,     2,     1,     0,     1,     1,     2,     1,
       0,     3,     1,     1,     1,     1,     1,     1,     1,     1,
       2,     1,     0,     1
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
#line 149 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name((yyvsp[0].valString), make_loc((yyloc))); }
#line 1649 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 3:
#line 153 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name((yyvsp[0].valString), make_loc((yyloc))); }
#line 1655 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 4:
#line 154 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("function", make_loc((yyloc))); }
#line 1661 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 5:
#line 155 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("model", make_loc((yyloc))); }
#line 1667 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 6:
#line 156 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("program", make_loc((yyloc))); }
#line 1673 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 7:
#line 160 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valPath) = new bi::Path((yyvsp[0].valName), nullptr, make_loc((yyloc))); }
#line 1679 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 8:
#line 161 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valPath) = new bi::Path((yyvsp[-2].valName), (yyvsp[0].valPath), make_loc((yyloc))); }
#line 1685 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 9:
#line 170 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BooleanLiteral((yyvsp[0].valBool), yytext, new bi::ModelReference(new bi::Name("Boolean")), make_loc((yyloc))); }
#line 1691 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 10:
#line 174 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::IntegerLiteral((yyvsp[0].valInt), yytext, new bi::ModelReference(new bi::Name("Integer")), make_loc((yyloc))); }
#line 1697 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 11:
#line 178 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::RealLiteral((yyvsp[0].valReal), yytext, new bi::ModelReference(new bi::Name("Real")), make_loc((yyloc))); }
#line 1703 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 12:
#line 182 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::StringLiteral((yyvsp[0].valString), yytext, new bi::ModelReference(new bi::Name("String")), make_loc((yyloc))); }
#line 1709 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 17:
#line 198 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ParenthesesExpression((yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 1715 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 19:
#line 203 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_empty(); }
#line 1721 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 20:
#line 207 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = (yyvsp[-1].valExpression); }
#line 1727 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 21:
#line 211 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracesExpression((yyvsp[-1].valStatement), make_loc((yyloc))); }
#line 1733 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 22:
#line 215 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracesExpression((yyvsp[-1].valStatement), make_loc((yyloc))); }
#line 1739 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 23:
#line 219 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracesExpression((yyvsp[-1].valStatement), make_loc((yyloc))); }
#line 1745 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 24:
#line 223 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracesExpression((yyvsp[-1].valStatement), make_loc((yyloc))); }
#line 1751 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 25:
#line 232 "parser.ypp" /* yacc.c:1661  */
    { raw << (yyvsp[0].valString); }
#line 1757 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 26:
#line 233 "parser.ypp" /* yacc.c:1661  */
    { raw << (yyvsp[0].valString); }
#line 1763 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 27:
#line 237 "parser.ypp" /* yacc.c:1661  */
    { raw.str(""); }
#line 1769 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 29:
#line 246 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::VarParameter((yyvsp[-2].valName), (yyvsp[0].valType), make_empty(), make_empty(), make_loc((yyloc))); }
#line 1775 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 30:
#line 247 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::VarParameter(new bi::Name(), (yyvsp[0].valType), make_empty(), make_empty(), make_loc((yyloc))); }
#line 1781 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 31:
#line 251 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = convert_ref((yyvsp[-3].valExpression), (yyvsp[-1].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1787 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 32:
#line 252 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = convert_ref((yyvsp[-3].valExpression), (yyvsp[-1].valExpression), make_empty(), make_loc((yyloc))); }
#line 1793 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 33:
#line 253 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = convert_ref((yyvsp[-1].valExpression), make_empty(), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1799 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 34:
#line 254 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = convert_ref((yyvsp[-1].valExpression), make_empty(), make_empty(), make_loc((yyloc))); }
#line 1805 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 35:
#line 258 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelReference((yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1811 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 36:
#line 262 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("<", make_loc((yyloc))); }
#line 1817 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 37:
#line 266 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("=", make_loc((yyloc))); }
#line 1823 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 38:
#line 270 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-4].valName), (yyvsp[-3].valExpression), (yyvsp[-2].valName), (yyvsp[-1].valType), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1829 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 39:
#line 271 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-3].valName), make_empty(), (yyvsp[-2].valName), (yyvsp[-1].valType), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1835 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 40:
#line 272 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-3].valName), make_empty(), (yyvsp[-2].valName), (yyvsp[-1].valType), make_empty(), make_loc((yyloc))); }
#line 1841 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 41:
#line 273 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-3].valName), make_empty(), (yyvsp[-2].valName), (yyvsp[-1].valType), make_empty(), make_loc((yyloc))); }
#line 1847 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 42:
#line 274 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-2].valName), (yyvsp[-1].valExpression), new bi::Name(), new bi::EmptyType(), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1853 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 43:
#line 275 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-1].valName), make_empty(), new bi::Name(), new bi::EmptyType(), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1859 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 44:
#line 276 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-1].valName), make_empty(), new bi::Name(), new bi::EmptyType(), make_empty(), make_loc((yyloc))); }
#line 1865 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 45:
#line 280 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valProg) = new bi::ProgParameter((yyvsp[-2].valName), (yyvsp[-1].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1871 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 46:
#line 281 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valProg) = new bi::ProgParameter((yyvsp[-2].valName), (yyvsp[-1].valExpression), make_empty(), make_loc((yyloc))); }
#line 1877 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 47:
#line 290 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::VarReference((yyvsp[0].valName), make_loc((yyloc))); }
#line 1883 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 48:
#line 294 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::FuncReference((yyvsp[-1].valName), (yyvsp[0].valExpression), bi::FUNCTION, make_loc((yyloc))); }
#line 1889 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 49:
#line 298 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelReference((yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1895 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 50:
#line 307 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ParenthesesType((yyvsp[-1].valType), make_loc((yyloc))); }
#line 1901 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 54:
#line 317 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::BracketsType((yyvsp[-1].valType), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1907 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 56:
#line 322 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::LambdaType((yyvsp[0].valType), make_loc((yyloc))); }
#line 1913 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 58:
#line 327 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::AssignableType((yyvsp[-1].valType), make_loc((yyloc))); }
#line 1919 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 60:
#line 332 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::RandomType((yyvsp[-2].valType), (yyvsp[0].valType), make_loc((yyloc))); }
#line 1925 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 65:
#line 355 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ParenthesesExpression((yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 1931 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 66:
#line 359 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::This(make_loc((yyloc))); }
#line 1937 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 71:
#line 370 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::Range((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1943 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 72:
#line 371 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::Index((yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1949 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 74:
#line 376 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ExpressionList((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1955 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 76:
#line 381 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracketsExpression((yyvsp[-3].valExpression), (yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 1961 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 79:
#line 390 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::Member((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1967 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 82:
#line 399 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("+", make_loc((yyloc))); }
#line 1973 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 83:
#line 400 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("-", make_loc((yyloc))); }
#line 1979 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 84:
#line 401 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("!", make_loc((yyloc))); }
#line 1985 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 86:
#line 406 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_unary((yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1991 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 87:
#line 410 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("*", make_loc((yyloc))); }
#line 1997 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 88:
#line 411 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("/", make_loc((yyloc))); }
#line 2003 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 90:
#line 416 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_binary((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2009 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 91:
#line 420 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("+", make_loc((yyloc))); }
#line 2015 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 92:
#line 421 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("-", make_loc((yyloc))); }
#line 2021 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 94:
#line 426 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_binary((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2027 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 95:
#line 430 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("<", make_loc((yyloc))); }
#line 2033 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 96:
#line 431 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name(">", make_loc((yyloc))); }
#line 2039 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 97:
#line 432 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("<=", make_loc((yyloc))); }
#line 2045 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 98:
#line 433 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name(">=", make_loc((yyloc))); }
#line 2051 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 100:
#line 438 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_binary((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2057 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 101:
#line 442 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("==", make_loc((yyloc))); }
#line 2063 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 102:
#line 443 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("!=", make_loc((yyloc))); }
#line 2069 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 104:
#line 448 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_binary((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2075 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 105:
#line 452 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("&&", make_loc((yyloc))); }
#line 2081 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 107:
#line 457 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_binary((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2087 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 108:
#line 461 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("||", make_loc((yyloc))); }
#line 2093 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 110:
#line 466 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_binary((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2099 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 111:
#line 470 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("->", make_loc((yyloc))); }
#line 2105 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 112:
#line 471 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("~>", make_loc((yyloc))); }
#line 2111 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 114:
#line 476 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_binary((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2117 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 115:
#line 480 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("<-", make_loc((yyloc))); }
#line 2123 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 116:
#line 481 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("<~", make_loc((yyloc))); }
#line 2129 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 118:
#line 486 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_assign((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2135 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 119:
#line 487 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::RandomInit((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2141 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 122:
#line 496 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ExpressionList((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2147 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 125:
#line 505 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_empty(); }
#line 2153 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 126:
#line 514 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = init_param((yyvsp[-2].valExpression), (yyvsp[0].valExpression)); }
#line 2159 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 129:
#line 520 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ExpressionList((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2165 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 131:
#line 525 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_empty(); }
#line 2171 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 132:
#line 529 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ParenthesesExpression((yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 2177 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 133:
#line 538 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::VarDeclaration(init_param((yyvsp[-3].valExpression), (yyvsp[-1].valExpression)), make_loc((yyloc))); }
#line 2183 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 134:
#line 539 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::VarDeclaration(dynamic_cast<bi::VarParameter*>((yyvsp[-1].valExpression)), make_loc((yyloc))); }
#line 2189 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 135:
#line 543 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::FuncDeclaration(dynamic_cast<bi::FuncParameter*>((yyvsp[0].valExpression)), make_loc((yyloc))); }
#line 2195 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 136:
#line 547 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::ModelDeclaration(dynamic_cast<bi::ModelParameter*>((yyvsp[0].valType)), make_loc((yyloc))); }
#line 2201 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 137:
#line 551 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::ProgDeclaration(dynamic_cast<bi::ProgParameter*>((yyvsp[0].valProg)), make_loc((yyloc))); }
#line 2207 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 138:
#line 555 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::ExpressionStatement((yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 2213 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 139:
#line 559 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Conditional((yyvsp[-3].valExpression), (yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2219 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 140:
#line 560 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Conditional((yyvsp[-3].valExpression), (yyvsp[-2].valExpression), new bi::BracesExpression((yyvsp[0].valStatement)), make_loc((yyloc))); }
#line 2225 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 141:
#line 561 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Conditional((yyvsp[-1].valExpression), (yyvsp[0].valExpression), make_empty(), make_loc((yyloc))); }
#line 2231 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 142:
#line 565 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Loop((yyvsp[-1].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2237 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 143:
#line 569 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Raw(new bi::Name("cpp"), raw.str(), make_loc((yyloc))); }
#line 2243 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 144:
#line 573 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Raw(new bi::Name("hpp"), raw.str(), make_loc((yyloc))); }
#line 2249 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 151:
#line 586 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2255 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 153:
#line 591 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2261 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 156:
#line 600 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2267 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 158:
#line 605 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2273 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 163:
#line 616 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2279 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 165:
#line 621 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2285 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 168:
#line 630 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2291 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 170:
#line 635 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2297 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 171:
#line 639 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Import((yyvsp[-1].valPath), compiler->import((yyvsp[-1].valPath)), make_loc((yyloc))); }
#line 2303 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 180:
#line 654 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2309 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 182:
#line 659 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2315 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 183:
#line 663 "parser.ypp" /* yacc.c:1661  */
    { compiler->setRoot((yyvsp[0].valStatement)); }
#line 2321 "parser.tab.cpp" /* yacc.c:1661  */
    break;


#line 2325 "parser.tab.cpp" /* yacc.c:1661  */
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
#line 666 "parser.ypp" /* yacc.c:1906  */

