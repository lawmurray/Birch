/**
 * @file
 */
#pragma once

#include "libubjpp/common/Parser.h"
#include "libubjpp/common/ParserState.hpp"

#include <regex>
#include <string>
#include <sstream>

namespace libubjpp {
/**
 * Tokenizer for JSON files.
 *
 * @ingroup libubjpp
 */
class JSONTokenizer {
public:
  /**
   * Constructor.
   *
   * @param data The JSON data.
   */
  JSONTokenizer(std::string& data);

  /**
   * Next token.
   *
   * @param[in,out] state Parser state. Updated to contain the value of the
   * next token (e.g. the actual string value for a STRING token, the actual
   * integer value for an INT64 token).
   *
   * @return Next token.
   */
  int next(ParserState* state);

private:
  /**
   * Input data.
   */
  std::string& data;

  /**
   * Input buffer for strings.
   */
  std::stringstream buf;

  /**
   * Token match results.
   */
  std::match_results<std::string::iterator> match;

  /*
   * Patterns.
   */
  std::regex regexInt, regexFrac, regexExp, regexTrue, regexFalse, regexNull;

  /*
   * Iterators over stream.
   */
  std::string::iterator begin, iter, end;
};
}
