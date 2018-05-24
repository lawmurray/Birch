/**
 * @file
 *
 * Flex externals.
 */
#pragma once

extern FILE* yyin;
extern char *yytext;
extern std::stringstream raw;

int yylex();
int yyparse();
void yyerror(const char *msg);
void yywarn(const char *msg);
void yylocation();
void yycount();
void yyreset();
