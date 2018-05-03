/**
 * @file
 *
 * The state of the parser, as seen from C code.
 */
#pragma once

#include <stdlib.h>
// ^ removes warning about implicit declaration of realloc() and free() when
//   %stack_size 0 used in Parser.y

typedef struct ParserState ParserState;

void push(ParserState* s);
void object(ParserState* s);
void array(ParserState* s);
void member(ParserState* s);
void element(ParserState* s);
void error(ParserState* s);
