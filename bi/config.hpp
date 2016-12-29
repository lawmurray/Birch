/**
 * @file
 *
 * Flex externals.
 */
#pragma once

#include <cstdio>

extern FILE* yyin;
extern char* yytext;

int yylex();
