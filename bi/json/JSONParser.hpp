/**
 * @file
 */
#pragma once

#include <regex>
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
   * @return Next token.
   */
  Integer next();

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
