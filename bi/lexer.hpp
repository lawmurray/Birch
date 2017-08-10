/**
 * @file
 *
 * Flex externals.
 */
#ifndef BI_LEXER_HPP
#define BI_LEXER_HPP

#include <cstdio>
#include <sstream>

extern FILE* yyin;
extern std::stringstream raw;

int yylex();
int yyparse();
void yyerror(const char *msg);
void yywarn(const char *msg);
void yylocation();
void yycount();
void yyreset();

#endif
