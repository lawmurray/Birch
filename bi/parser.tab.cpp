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
    RIGHT_OP = 277,
    LEFT_OP = 278,
    PUSH_OP = 279,
    PULL_OP = 280,
    AND_OP = 281,
    OR_OP = 282,
    LE_OP = 283,
    GE_OP = 284,
    EQ_OP = 285,
    NE_OP = 286,
    RANGE_OP = 287
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

#line 161 "parser.tab.cpp" /* yacc.c:355  */
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

#line 192 "parser.tab.cpp" /* yacc.c:358  */
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

#line 257 "parser.tab.cpp" /* yacc.c:359  */

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
#define YYFINAL  85
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   221

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  52
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  96
/* YYNRULES -- Number of rules.  */
#define YYNRULES  179
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  246

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   287

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    47,     2,     2,     2,     2,     2,     2,
      34,    35,    48,    45,    51,    46,    33,    49,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    38,    39,
      40,    41,    50,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    42,     2,    43,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    36,     2,    37,    44,     2,     2,     2,
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
      25,    26,    27,    28,    29,    30,    31,    32
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   147,   147,   151,   152,   153,   154,   158,   159,   168,
     172,   176,   180,   184,   185,   186,   187,   196,   200,   204,
     208,   212,   221,   222,   226,   226,   235,   236,   237,   241,
     242,   243,   244,   248,   249,   253,   257,   261,   262,   263,
     264,   265,   266,   267,   271,   272,   281,   285,   289,   290,
     291,   300,   304,   308,   309,   310,   314,   315,   319,   324,
     333,   334,   338,   342,   346,   347,   348,   349,   353,   354,
     358,   362,   363,   367,   368,   372,   373,   374,   378,   379,
     383,   384,   388,   389,   393,   394,   398,   399,   403,   407,
     408,   412,   413,   414,   415,   419,   420,   424,   425,   429,
     430,   434,   438,   439,   443,   447,   448,   452,   456,   457,
     461,   462,   466,   470,   471,   472,   476,   480,   481,   485,
     489,   490,   499,   500,   504,   505,   509,   510,   514,   523,
     524,   528,   532,   536,   540,   544,   545,   546,   550,   554,
     558,   562,   563,   564,   565,   566,   570,   571,   575,   576,
     580,   584,   585,   589,   590,   594,   595,   596,   600,   601,
     605,   606,   610,   614,   615,   619,   620,   624,   628,   629,
     630,   631,   632,   633,   634,   638,   639,   643,   644,   648
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
  "REAL_LITERAL", "STRING_LITERAL", "RIGHT_OP", "LEFT_OP", "PUSH_OP",
  "PULL_OP", "AND_OP", "OR_OP", "LE_OP", "GE_OP", "EQ_OP", "NE_OP",
  "RANGE_OP", "'.'", "'('", "')'", "'{'", "'}'", "':'", "';'", "'<'",
  "'='", "'['", "']'", "'~'", "'+'", "'-'", "'!'", "'*'", "'/'", "'>'",
  "','", "$accept", "name", "path_name", "path", "bool_literal",
  "int_literal", "real_literal", "string_literal", "literal", "parens",
  "braces", "func_braces", "model_braces", "prog_braces", "raw",
  "double_braces", "$@1", "var_parameter", "func_parameter", "base",
  "base_less_operator", "base_equal_operator", "model_parameter",
  "prog_parameter", "var_reference", "func_reference", "model_reference",
  "parens_type", "any_type", "primary_type", "random_type", "list_type",
  "type", "reference_expression", "parens_expression", "this_expression",
  "primary_expression", "brackets_expression", "member_operator",
  "member_expression", "parameter_expression", "unary_operator",
  "unary_expression", "multiplicative_operator",
  "multiplicative_expression", "additive_operator", "additive_expression",
  "range_operator", "range_expression", "relational_operator",
  "relational_expression", "equality_operator", "equality_expression",
  "logical_and_operator", "logical_and_expression", "logical_or_operator",
  "logical_or_expression", "push_operator", "push_expression",
  "assignment_operator", "random_operator", "assignment_expression",
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
     285,   286,   287,    46,    40,    41,   123,   125,    58,    59,
      60,    61,    91,    93,   126,    43,    45,    33,    42,    47,
      62,    44
};
# endif

#define YYPACT_NINF -203

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-203)))

#define YYTABLE_NINF -1

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
      41,   106,    39,    39,    88,    57,    57,  -203,    46,    76,
      18,  -203,  -203,  -203,  -203,  -203,  -203,  -203,    41,  -203,
    -203,    84,  -203,  -203,  -203,  -203,    83,    81,    98,  -203,
     102,  -203,  -203,  -203,  -203,  -203,  -203,    88,  -203,  -203,
    -203,    30,  -203,  -203,  -203,  -203,  -203,  -203,  -203,  -203,
    -203,  -203,  -203,  -203,   103,  -203,   129,  -203,   132,  -203,
      82,   -17,  -203,    37,   117,   137,    43,    51,   118,  -203,
      79,  -203,  -203,  -203,  -203,    46,    -4,  -203,  -203,  -203,
    -203,    46,    88,  -203,  -203,  -203,   106,  -203,    56,    85,
      88,    31,  -203,  -203,  -203,    77,  -203,    39,    39,   133,
    -203,    88,  -203,   155,  -203,  -203,  -203,    88,  -203,  -203,
    -203,    88,    88,  -203,  -203,  -203,  -203,    88,  -203,  -203,
      88,  -203,    88,  -203,  -203,    88,    88,  -203,  -203,  -203,
      88,    88,  -203,    88,   132,    15,  -203,  -203,   161,   127,
    -203,  -203,   143,    88,  -203,   145,   142,  -203,   157,  -203,
     118,   147,    15,  -203,  -203,  -203,   148,  -203,  -203,  -203,
      31,  -203,   149,  -203,    39,   145,    89,   146,  -203,   144,
     145,  -203,  -203,    82,   109,  -203,    37,   117,   137,  -203,
    -203,  -203,  -203,   101,   145,   145,   151,  -203,  -203,  -203,
    -203,  -203,  -203,    15,  -203,   154,  -203,   141,    46,  -203,
     150,  -203,  -203,    88,    56,  -203,  -203,  -203,    15,   158,
    -203,  -203,  -203,   152,  -203,  -203,  -203,  -203,  -203,  -203,
    -203,   156,   156,  -203,  -203,  -203,  -203,  -203,  -203,  -203,
    -203,  -203,  -203,  -203,  -203,    15,   176,  -203,    15,  -203,
     159,     7,  -203,  -203,  -203,  -203
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
     178,     0,     0,     0,     0,     0,     0,     2,     0,     0,
       0,   171,   172,   174,   173,   169,   170,   168,   175,   177,
     179,     0,     6,     5,     4,     3,     7,     0,     0,   133,
       0,   132,    63,     9,    10,    11,    12,     0,    75,    76,
      77,    46,    13,    14,    15,    16,    64,    73,   131,    60,
      61,    66,    65,    67,    68,    71,    74,    78,     0,    82,
      86,    89,    95,    99,   102,   105,   108,   113,   117,   119,
       0,    24,   139,   140,    52,     0,    50,    53,    54,    55,
      28,     0,     0,   130,   176,     1,     0,   167,   127,     0,
     121,   161,    43,    35,    36,     0,    42,     0,     0,     0,
      47,     0,    70,     0,    79,    80,    81,     0,    88,    84,
      85,     0,     0,    93,    94,    91,    92,     0,    97,    98,
       0,   101,     0,   107,   104,     0,     0,   110,   111,   112,
       0,     0,   116,     0,     0,   154,    32,    31,     0,    56,
      58,    59,     0,     0,    48,    27,     0,     8,   123,   124,
     126,     0,   166,    45,    44,   120,     0,   155,   156,   157,
     158,   160,     0,    41,     0,    34,     0,     0,    62,     0,
      46,    72,    83,    87,    90,    96,   100,   103,   106,   109,
     114,   115,   118,     0,     0,     0,     0,   141,   142,   143,
     144,   145,   150,   151,   153,     0,    22,     0,     0,    51,
       0,    26,   129,     0,     0,   128,   162,   163,   165,     0,
      17,   159,    20,     0,    33,    39,    38,    40,    69,    30,
      29,     0,     0,   134,   152,    19,    25,    23,    57,    49,
     122,   125,   164,    21,    37,   149,   137,   138,   146,   148,
       0,     0,   147,    18,   135,   136
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -203,     0,  -203,   108,  -203,  -203,  -203,  -203,  -203,   -26,
    -202,    14,   -84,  -203,  -203,   193,  -203,     1,  -203,   -86,
     105,  -203,  -203,  -203,  -203,  -203,  -203,  -203,  -203,     2,
       4,  -203,  -203,  -203,  -203,  -203,  -203,   100,  -203,  -203,
     -49,  -203,    94,  -203,    95,  -203,    96,  -203,    90,  -203,
      91,  -203,    87,  -203,    92,  -203,  -203,  -203,    86,  -203,
    -203,   -76,    60,    80,     3,  -203,    10,  -203,  -203,  -203,
     -74,   -70,  -203,  -203,  -203,   -25,  -203,    13,     5,  -136,
     -23,  -203,  -203,    25,  -203,  -203,    59,  -203,    12,  -203,
    -203,  -203,  -203,   203,  -203,  -203
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,    41,    26,    27,    42,    43,    44,    45,    46,   100,
     236,   137,    96,   154,   197,    72,   138,    47,    48,   166,
      97,    98,    31,    29,    49,    50,    77,    78,    79,   139,
     140,   141,   142,    51,    52,    53,    54,    55,   103,    56,
      57,    58,    59,   107,    60,   111,    61,   112,    62,   117,
      63,   120,    64,   122,    65,   125,    66,   126,    67,   130,
     131,    68,   133,    69,   186,   156,   149,   150,   151,    89,
      11,    12,    13,    14,   187,   188,   189,   190,   191,   192,
     239,   240,   193,   194,   195,   160,   161,   162,   207,   208,
     209,    17,    18,    19,    20,    21
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint8 yytable[] =
{
       9,    10,    28,    30,    95,    16,   146,    70,    76,   104,
      80,   163,   167,    15,   184,   108,   206,   157,     9,    10,
     237,   158,   184,    16,   185,     5,     6,    32,   109,   110,
      90,    15,     7,    33,    34,    35,    36,     4,   143,   244,
      99,    82,     6,   235,     1,     2,     3,     4,     7,    37,
     144,     5,     6,     8,   180,   181,     7,    83,     7,    74,
      38,    39,    40,     7,    90,   113,   114,   123,    81,     8,
     124,    71,   206,     7,   127,    76,   128,   115,   213,     8,
      75,    76,   216,   145,    85,   183,   157,   116,     9,   148,
     158,     9,    10,   155,     8,   129,   159,   165,   165,   238,
      32,   134,   238,   170,   169,     7,    33,    34,    35,    36,
      22,    23,    24,    91,    81,   135,    86,    93,   136,   201,
      87,   152,    37,    25,   153,    91,     8,   230,   215,   234,
     105,   106,    88,    38,    39,    40,    90,   135,    91,   214,
     219,    92,    93,    94,    32,   101,   200,   118,   119,     7,
      33,    34,    35,    36,   109,   110,   226,   227,   221,   222,
       9,    10,   102,   121,   165,   159,    37,    32,   168,   132,
       8,   198,     7,    33,    34,    35,    36,   196,   199,    90,
     203,   202,   205,   210,   241,   217,   212,   218,    91,    37,
     223,   225,   235,   229,   147,   233,   243,   220,    76,    73,
     164,   172,   228,   171,     9,   148,   173,   175,   174,   177,
     204,   176,   179,   182,   231,   242,   245,   178,   224,   211,
     232,    84
};

static const yytype_uint8 yycheck[] =
{
       0,     0,     2,     3,    30,     0,    82,     4,     8,    58,
       8,    95,    98,     0,     7,    32,   152,    91,    18,    18,
     222,    91,     7,    18,     9,    10,    11,    12,    45,    46,
      34,    18,    17,    18,    19,    20,    21,     6,    42,   241,
      37,    23,    11,    36,     3,     4,     5,     6,    17,    34,
      76,    10,    11,    38,   130,   131,    17,    39,    17,    13,
      45,    46,    47,    17,    34,    28,    29,    24,    38,    38,
      27,    14,   208,    17,    23,    75,    25,    40,   164,    38,
      34,    81,   166,    81,     0,   134,   160,    50,    88,    88,
     160,    91,    91,    90,    38,    44,    91,    97,    98,   235,
      12,    22,   238,   103,   101,    17,    18,    19,    20,    21,
       4,     5,     6,    36,    38,    36,    33,    40,    39,   145,
      39,    36,    34,    17,    39,    36,    38,   203,    39,   213,
      48,    49,    34,    45,    46,    47,    34,    36,    36,   165,
      39,    39,    40,    41,    12,    42,   143,    30,    31,    17,
      18,    19,    20,    21,    45,    46,    15,    16,   184,   185,
     160,   160,    33,    26,   164,   160,    34,    12,    35,    51,
      38,    44,    17,    18,    19,    20,    21,    16,    35,    34,
      23,    39,    35,    35,     8,    39,    37,    43,    36,    34,
      39,    37,    36,    43,    86,    37,    37,   183,   198,     6,
      95,   107,   198,   103,   204,   204,   111,   117,   112,   122,
     150,   120,   126,   133,   204,   238,   241,   125,   193,   160,
     208,    18
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,     5,     6,    10,    11,    17,    38,    53,
      69,   122,   123,   124,   125,   129,   130,   143,   144,   145,
     146,   147,     4,     5,     6,    17,    54,    55,    53,    75,
      53,    74,    12,    18,    19,    20,    21,    34,    45,    46,
      47,    53,    56,    57,    58,    59,    60,    69,    70,    76,
      77,    85,    86,    87,    88,    89,    91,    92,    93,    94,
      96,    98,   100,   102,   104,   106,   108,   110,   113,   115,
     116,    14,    67,    67,    13,    34,    53,    78,    79,    80,
      81,    38,    23,    39,   145,     0,    33,    39,    34,   121,
      34,    36,    39,    40,    41,    61,    64,    72,    73,   116,
      61,    42,    33,    90,    92,    48,    49,    95,    32,    45,
      46,    97,    99,    28,    29,    40,    50,   101,    30,    31,
     103,    26,   105,    24,    27,   107,   109,    23,    25,    44,
     111,   112,    51,   114,    22,    36,    39,    63,    68,    81,
      82,    83,    84,    42,    61,    81,   113,    55,    69,   118,
     119,   120,    36,    39,    65,   116,   117,   122,   123,   130,
     137,   138,   139,    64,    72,    53,    71,    71,    35,   116,
      53,    89,    94,    96,    98,   100,   102,   104,   106,   110,
     113,   113,   115,    92,     7,     9,   116,   126,   127,   128,
     129,   130,   131,   134,   135,   136,    16,    66,    44,    35,
     116,    61,    39,    23,   114,    35,   131,   140,   141,   142,
      35,   138,    37,    71,    61,    39,    64,    39,    43,    39,
      63,    61,    61,    39,   135,    37,    15,    16,    82,    43,
     113,   118,   140,    37,    64,    36,    62,    62,   131,   132,
     133,     8,   132,    37,    62,   127
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    52,    53,    54,    54,    54,    54,    55,    55,    56,
      57,    58,    59,    60,    60,    60,    60,    61,    62,    63,
      64,    65,    66,    66,    68,    67,    69,    69,    69,    70,
      70,    70,    70,    71,    71,    72,    73,    74,    74,    74,
      74,    74,    74,    74,    75,    75,    76,    77,    78,    78,
      78,    79,    80,    81,    81,    81,    82,    82,    83,    84,
      85,    85,    86,    87,    88,    88,    88,    88,    89,    89,
      90,    91,    91,    92,    92,    93,    93,    93,    94,    94,
      95,    95,    96,    96,    97,    97,    98,    98,    99,   100,
     100,   101,   101,   101,   101,   102,   102,   103,   103,   104,
     104,   105,   106,   106,   107,   108,   108,   109,   110,   110,
     111,   111,   112,   113,   113,   113,   114,   115,   115,   116,
     117,   117,   118,   118,   119,   119,   120,   120,   121,   122,
     122,   123,   124,   125,   126,   127,   127,   127,   128,   129,
     130,   131,   131,   131,   131,   131,   132,   132,   133,   133,
     134,   135,   135,   136,   136,   137,   137,   137,   138,   138,
     139,   139,   140,   141,   141,   142,   142,   143,   144,   144,
     144,   144,   144,   144,   144,   145,   145,   146,   146,   147
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     1,     1,     1,     1,     1,     3,     1,
       1,     1,     1,     1,     1,     1,     1,     3,     3,     3,
       3,     3,     1,     2,     0,     4,     4,     3,     2,     4,
       4,     2,     2,     2,     1,     1,     1,     5,     4,     4,
       4,     3,     2,     2,     3,     3,     1,     2,     2,     4,
       1,     3,     1,     1,     1,     1,     1,     3,     1,     1,
       1,     1,     3,     1,     1,     1,     1,     1,     1,     4,
       1,     1,     3,     1,     1,     1,     1,     1,     1,     2,
       1,     1,     1,     3,     1,     1,     1,     3,     1,     1,
       3,     1,     1,     1,     1,     1,     3,     1,     1,     1,
       3,     1,     1,     3,     1,     1,     3,     1,     1,     3,
       1,     1,     1,     1,     3,     3,     1,     1,     3,     1,
       1,     0,     3,     1,     1,     3,     1,     0,     3,     4,
       2,     2,     2,     2,     2,     5,     5,     3,     3,     2,
       2,     1,     1,     1,     1,     1,     1,     2,     1,     0,
       1,     1,     2,     1,     0,     1,     1,     1,     1,     2,
       1,     0,     1,     1,     2,     1,     0,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     2,     1,     0,     1
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
#line 147 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name((yyvsp[0].valString), make_loc((yyloc))); }
#line 1641 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 3:
#line 151 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name((yyvsp[0].valString), make_loc((yyloc))); }
#line 1647 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 4:
#line 152 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("function", make_loc((yyloc))); }
#line 1653 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 5:
#line 153 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("model", make_loc((yyloc))); }
#line 1659 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 6:
#line 154 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("program", make_loc((yyloc))); }
#line 1665 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 7:
#line 158 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valPath) = new bi::Path((yyvsp[0].valName), nullptr, make_loc((yyloc))); }
#line 1671 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 8:
#line 159 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valPath) = new bi::Path((yyvsp[-2].valName), (yyvsp[0].valPath), make_loc((yyloc))); }
#line 1677 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 9:
#line 168 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BooleanLiteral((yyvsp[0].valBool), yytext, new bi::ModelReference(new bi::Name("Boolean")), make_loc((yyloc))); }
#line 1683 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 10:
#line 172 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::IntegerLiteral((yyvsp[0].valInt), yytext, new bi::ModelReference(new bi::Name("Integer")), make_loc((yyloc))); }
#line 1689 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 11:
#line 176 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::RealLiteral((yyvsp[0].valReal), yytext, new bi::ModelReference(new bi::Name("Real")), make_loc((yyloc))); }
#line 1695 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 12:
#line 180 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::StringLiteral((yyvsp[0].valString), yytext, new bi::ModelReference(new bi::Name("String")), make_loc((yyloc))); }
#line 1701 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 17:
#line 196 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ParenthesesExpression((yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 1707 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 18:
#line 200 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracesExpression((yyvsp[-1].valStatement), make_loc((yyloc))); }
#line 1713 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 19:
#line 204 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracesExpression((yyvsp[-1].valStatement), make_loc((yyloc))); }
#line 1719 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 20:
#line 208 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracesExpression((yyvsp[-1].valStatement), make_loc((yyloc))); }
#line 1725 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 21:
#line 212 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracesExpression((yyvsp[-1].valStatement), make_loc((yyloc))); }
#line 1731 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 22:
#line 221 "parser.ypp" /* yacc.c:1661  */
    { raw << (yyvsp[0].valString); }
#line 1737 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 23:
#line 222 "parser.ypp" /* yacc.c:1661  */
    { raw << (yyvsp[0].valString); }
#line 1743 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 24:
#line 226 "parser.ypp" /* yacc.c:1661  */
    { raw.str(""); }
#line 1749 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 26:
#line 235 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::VarParameter((yyvsp[-3].valName), (yyvsp[-1].valType), (yyvsp[0].valExpression), make_empty(), make_loc((yyloc))); }
#line 1755 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 27:
#line 236 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::VarParameter((yyvsp[-2].valName), (yyvsp[0].valType), make_empty(), make_empty(), make_loc((yyloc))); }
#line 1761 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 28:
#line 237 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::VarParameter(new bi::Name(), (yyvsp[0].valType), make_empty(), make_empty(), make_loc((yyloc))); }
#line 1767 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 29:
#line 241 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = convert_ref((yyvsp[-3].valExpression), (yyvsp[-1].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1773 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 30:
#line 242 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = convert_ref((yyvsp[-3].valExpression), (yyvsp[-1].valExpression), make_empty(), make_loc((yyloc))); }
#line 1779 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 31:
#line 243 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = convert_ref((yyvsp[-1].valExpression), make_empty(), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1785 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 32:
#line 244 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = convert_ref((yyvsp[-1].valExpression), make_empty(), make_empty(), make_loc((yyloc))); }
#line 1791 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 33:
#line 248 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelReference((yyvsp[-1].valName), (yyvsp[0].valExpression), make_empty(), make_loc((yyloc))); }
#line 1797 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 34:
#line 249 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelReference((yyvsp[0].valName), make_empty(), make_empty(), make_loc((yyloc))); }
#line 1803 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 35:
#line 253 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("<", make_loc((yyloc))); }
#line 1809 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 36:
#line 257 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("=", make_loc((yyloc))); }
#line 1815 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 37:
#line 261 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-4].valName), (yyvsp[-3].valExpression), (yyvsp[-2].valName), (yyvsp[-1].valType), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1821 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 38:
#line 262 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-3].valName), make_empty(), (yyvsp[-2].valName), (yyvsp[-1].valType), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1827 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 39:
#line 263 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-3].valName), make_empty(), (yyvsp[-2].valName), (yyvsp[-1].valType), make_empty(), make_loc((yyloc))); }
#line 1833 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 40:
#line 264 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-3].valName), make_empty(), (yyvsp[-2].valName), (yyvsp[-1].valType), make_empty(), make_loc((yyloc))); }
#line 1839 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 41:
#line 265 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-2].valName), (yyvsp[-1].valExpression), new bi::Name(), new bi::EmptyType(), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1845 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 42:
#line 266 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-1].valName), make_empty(), new bi::Name(), new bi::EmptyType(), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1851 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 43:
#line 267 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelParameter((yyvsp[-1].valName), make_empty(), new bi::Name(), new bi::EmptyType(), make_empty(), make_loc((yyloc))); }
#line 1857 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 44:
#line 271 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valProg) = new bi::ProgParameter((yyvsp[-2].valName), (yyvsp[-1].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1863 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 45:
#line 272 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valProg) = new bi::ProgParameter((yyvsp[-2].valName), (yyvsp[-1].valExpression), make_empty(), make_loc((yyloc))); }
#line 1869 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 46:
#line 281 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::VarReference((yyvsp[0].valName), make_loc((yyloc))); }
#line 1875 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 47:
#line 285 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::FuncReference((yyvsp[-1].valName), (yyvsp[0].valExpression), bi::FUNCTION, make_loc((yyloc))); }
#line 1881 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 48:
#line 289 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelReference((yyvsp[-1].valName), (yyvsp[0].valExpression), make_empty(), make_loc((yyloc))); }
#line 1887 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 49:
#line 290 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelReference((yyvsp[-3].valName), make_empty(), (yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 1893 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 50:
#line 291 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ModelReference((yyvsp[0].valName), make_empty(), make_empty(), make_loc((yyloc))); }
#line 1899 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 51:
#line 300 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::ParenthesesType((yyvsp[-1].valType), make_loc((yyloc))); }
#line 1905 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 52:
#line 304 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::EmptyType(); }
#line 1911 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 57:
#line 315 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valType) = new bi::RandomType((yyvsp[-2].valType), (yyvsp[0].valType), make_loc((yyloc))); }
#line 1917 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 62:
#line 338 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ParenthesesExpression((yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 1923 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 63:
#line 342 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::This(make_loc((yyloc))); }
#line 1929 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 69:
#line 354 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::BracketsExpression((yyvsp[-3].valExpression), (yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 1935 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 72:
#line 363 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::Member((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1941 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 75:
#line 372 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("+", make_loc((yyloc))); }
#line 1947 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 76:
#line 373 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("-", make_loc((yyloc))); }
#line 1953 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 77:
#line 374 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("!", make_loc((yyloc))); }
#line 1959 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 79:
#line 379 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_unary((yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1965 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 80:
#line 383 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("*", make_loc((yyloc))); }
#line 1971 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 81:
#line 384 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("/", make_loc((yyloc))); }
#line 1977 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 83:
#line 389 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_binary((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 1983 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 84:
#line 393 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("+", make_loc((yyloc))); }
#line 1989 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 85:
#line 394 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("-", make_loc((yyloc))); }
#line 1995 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 87:
#line 399 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_binary((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2001 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 90:
#line 408 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::Range((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2007 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 91:
#line 412 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("<", make_loc((yyloc))); }
#line 2013 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 92:
#line 413 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name(">", make_loc((yyloc))); }
#line 2019 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 93:
#line 414 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("<=", make_loc((yyloc))); }
#line 2025 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 94:
#line 415 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name(">=", make_loc((yyloc))); }
#line 2031 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 96:
#line 420 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_binary((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2037 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 97:
#line 424 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("==", make_loc((yyloc))); }
#line 2043 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 98:
#line 425 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("!=", make_loc((yyloc))); }
#line 2049 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 100:
#line 430 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_binary((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2055 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 101:
#line 434 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("&&", make_loc((yyloc))); }
#line 2061 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 103:
#line 439 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_binary((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2067 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 104:
#line 443 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("||", make_loc((yyloc))); }
#line 2073 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 106:
#line 448 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_binary((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2079 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 107:
#line 452 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("~>", make_loc((yyloc))); }
#line 2085 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 109:
#line 457 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_binary((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2091 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 110:
#line 461 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("<-", make_loc((yyloc))); }
#line 2097 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 111:
#line 462 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("<~", make_loc((yyloc))); }
#line 2103 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 112:
#line 466 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valName) = new bi::Name("~", make_loc((yyloc))); }
#line 2109 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 114:
#line 471 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_assign((yyvsp[-2].valExpression), (yyvsp[-1].valName), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2115 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 115:
#line 472 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::RandomInit((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2121 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 118:
#line 481 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ExpressionList((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2127 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 121:
#line 490 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_empty(); }
#line 2133 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 122:
#line 499 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = init_param((yyvsp[-2].valExpression), (yyvsp[0].valExpression)); }
#line 2139 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 125:
#line 505 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ExpressionList((yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2145 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 127:
#line 510 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = make_empty(); }
#line 2151 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 128:
#line 514 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valExpression) = new bi::ParenthesesExpression((yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 2157 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 129:
#line 523 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::VarDeclaration(init_param((yyvsp[-3].valExpression), (yyvsp[-1].valExpression)), make_loc((yyloc))); }
#line 2163 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 130:
#line 524 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::VarDeclaration(dynamic_cast<bi::VarParameter*>((yyvsp[-1].valExpression)), make_loc((yyloc))); }
#line 2169 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 131:
#line 528 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::FuncDeclaration(dynamic_cast<bi::FuncParameter*>((yyvsp[0].valExpression)), make_loc((yyloc))); }
#line 2175 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 132:
#line 532 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::ModelDeclaration(dynamic_cast<bi::ModelParameter*>((yyvsp[0].valType)), make_loc((yyloc))); }
#line 2181 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 133:
#line 536 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::ProgDeclaration(dynamic_cast<bi::ProgParameter*>((yyvsp[0].valProg)), make_loc((yyloc))); }
#line 2187 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 134:
#line 540 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::ExpressionStatement((yyvsp[-1].valExpression), make_loc((yyloc))); }
#line 2193 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 135:
#line 544 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Conditional((yyvsp[-3].valExpression), (yyvsp[-2].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2199 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 136:
#line 545 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Conditional((yyvsp[-3].valExpression), (yyvsp[-2].valExpression), new bi::BracesExpression((yyvsp[0].valStatement)), make_loc((yyloc))); }
#line 2205 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 137:
#line 546 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Conditional((yyvsp[-1].valExpression), (yyvsp[0].valExpression), make_empty(), make_loc((yyloc))); }
#line 2211 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 138:
#line 550 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Loop((yyvsp[-1].valExpression), (yyvsp[0].valExpression), make_loc((yyloc))); }
#line 2217 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 139:
#line 554 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Raw(new bi::Name("cpp"), raw.str(), make_loc((yyloc))); }
#line 2223 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 140:
#line 558 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Raw(new bi::Name("hpp"), raw.str(), make_loc((yyloc))); }
#line 2229 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 147:
#line 571 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2235 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 149:
#line 576 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2241 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 152:
#line 585 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2247 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 154:
#line 590 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2253 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 159:
#line 601 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2259 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 161:
#line 606 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2265 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 164:
#line 615 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2271 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 166:
#line 620 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2277 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 167:
#line 624 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::Import((yyvsp[-1].valPath), compiler->import((yyvsp[-1].valPath)), make_loc((yyloc))); }
#line 2283 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 176:
#line 639 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::StatementList((yyvsp[-1].valStatement), (yyvsp[0].valStatement), make_loc((yyloc))); }
#line 2289 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 178:
#line 644 "parser.ypp" /* yacc.c:1661  */
    { (yyval.valStatement) = new bi::EmptyStatement(); }
#line 2295 "parser.tab.cpp" /* yacc.c:1661  */
    break;

  case 179:
#line 648 "parser.ypp" /* yacc.c:1661  */
    { compiler->setRoot((yyvsp[0].valStatement)); }
#line 2301 "parser.tab.cpp" /* yacc.c:1661  */
    break;


#line 2305 "parser.tab.cpp" /* yacc.c:1661  */
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
#line 651 "parser.ypp" /* yacc.c:1906  */

