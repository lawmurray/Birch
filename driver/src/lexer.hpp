/**
 * @file
 *
 * Flex externals.
 */
#pragma once

extern FILE* yyin;
extern char *yytext;
extern std::stringstream raw;
extern int yylineno;

int yylex();
int yyparse();
void yyerror(const char *);
void yywarn(const char *);
void yylocation();
void yycount();
void yyreset();
void yyrestart(FILE*);
int yylex_destroy();
