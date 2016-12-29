/**
 * @file
 *
 * Flex externals.
 */
#ifndef BI_LEXER_HPP
#define BI_LEXER_HPP

#include <cstdio>

extern FILE* yyin;

int yylex();
int yyparse();
void yyerror(const char *msg);
void yywarn(const char *msg);
void yylocation();
void yycount();
void yyreset();

#endif
