/**
 * @file
 */
#pragma once

#include <regex>
#include <string>
#include <sstream>

namespace bi {
namespace type {
/**
 * Parser for JSON files.
 */
class JSONParser : public Parser {
protected:
  /**
   * Constructor.
   */
  JSONParser();

  /**
   * Next token.
   *
   * @param[in,out] state Parser state. Updated to contain the value of the
   * next token (e.g. the actual string value for a STRING token, the actual
   * integer value for an INT64 token).
   *
   * @return Next token.
   */
  virtual Integer next();

private:
  /**
   * Input data.
   */
  std::string data;

  /**
   * Input buffer for strings.
   */
  std::stringstream buf;

  /**
   * Token match results.
   */
  std::match_results<std::string::const_iterator> match;

  /*
   * Patterns.
   */
  std::regex regexInt, regexFrac, regexExp, regexTrue, regexFalse, regexNull;

  /*
   * Iterators over stream.
   */
  std::string::const_iterator begin, iter, end;
};
}
}
