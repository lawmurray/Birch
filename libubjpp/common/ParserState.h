/**
 * @file
 *
 * The state of the parser, as seen from C code.
 */
#pragma once

typedef struct ParserState ParserState;

void push(ParserState* s);
void object(ParserState* s);
void array(ParserState* s);
void member(ParserState* s);
void element(ParserState* s);
void error(ParserState* s);
