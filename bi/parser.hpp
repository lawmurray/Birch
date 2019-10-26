/* A Bison parser, made by GNU Bison 3.4.2.  */

/* Skeleton interface for Bison GLR parsers in C

   Copyright (C) 2002-2015, 2018-2019 Free Software Foundation, Inc.

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

#ifndef YY_YY_BI_PARSER_HPP_INCLUDED
# define YY_YY_BI_PARSER_HPP_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif
/* "%code requires" blocks.  */
#line 1 "bi/parser.ypp"

  #include "bi/lexer.hpp"
  #include "bi/build/Compiler.hpp"

#line 49 "bi/parser.hpp"

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    PROGRAM = 258,
    CLASS = 259,
    TYPE = 260,
    FUNCTION = 261,
    FIBER = 262,
    OPERATOR = 263,
    AUTO = 264,
    IF = 265,
    ELSE = 266,
    FOR = 267,
    IN = 268,
    WHILE = 269,
    DO = 270,
    ASSERT = 271,
    RETURN = 272,
    YIELD = 273,
    INSTANTIATED = 274,
    CPP = 275,
    HPP = 276,
    THIS = 277,
    SUPER = 278,
    GLOBAL = 279,
    PARALLEL = 280,
    DYNAMIC = 281,
    FINAL = 282,
    ABSTRACT = 283,
    NIL = 284,
    DOUBLE_BRACE_OPEN = 285,
    DOUBLE_BRACE_CLOSE = 286,
    NAME = 287,
    BOOL_LITERAL = 288,
    INT_LITERAL = 289,
    REAL_LITERAL = 290,
    STRING_LITERAL = 291,
    LEFT_OP = 292,
    RIGHT_OP = 293,
    LEFT_TILDE_OP = 294,
    RIGHT_TILDE_OP = 295,
    LEFT_QUERY_OP = 296,
    AND_OP = 297,
    OR_OP = 298,
    LE_OP = 299,
    GE_OP = 300,
    EQ_OP = 301,
    NE_OP = 302,
    RANGE_OP = 303
  };
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 71 "bi/parser.ypp"

  bool valBool;
  int32_t valInt;
  double valReal;
  const char* valString;
  
  bi::Annotation valAnnotation;
  bi::Name* valName;
  bi::Expression* valExpression;
  bi::Type* valType;
  bi::Statement* valStatement;

#line 122 "bi/parser.hpp"

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

#endif /* !YY_YY_BI_PARSER_HPP_INCLUDED  */
