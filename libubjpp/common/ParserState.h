/**
 * @file
 *
 * The state of the parser, as seen from C code.
 */
#pragma once

typedef struct ParserState ParserState;

void member(ParserState* s);
void element(ParserState* s);
void error(ParserState* s);
