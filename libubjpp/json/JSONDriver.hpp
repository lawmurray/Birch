/**
 * @file
 */
#pragma once

#include "libubjpp/value.hpp"

#include <string>

namespace libubjpp {
/**
 * Driver for parsing JSON files.
 *
 * @ingroup libubjpp
 */
class JSONDriver {
public:
  /**
   * Parse a file.
   *
   * @param file The file name.
   */
  boost::optional<value> parse(const std::string& file);

  /**
   * Parse a string.
   *
   * @param data The string.
   */
  boost::optional<value> parseString(const std::string& data);
};
}
