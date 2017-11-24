/**
 * @file
 *
 * The state of the parser, as seen from C++ code.
 */
#pragma once

#include "libubjpp/value.hpp"

#include <stack>

struct ParserState {
  /**
   * Push a value onto the stack (used by tokenizer).
   */
  void push(const libubjpp::value& value);

  /**
   * Get the root value of the data.
   */
  libubjpp::value root();

  void member();
  void element();

private:
  /**
   * Stack of values.
   */
  std::stack<libubjpp::value> values;
};

/**
 * Reduce an object member.
 */
extern "C" void member(ParserState* s);

/**
 * Reduce an array element.
 */
extern "C" void element(ParserState* s);
