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
   * Parse a stream.
   *
   * @param stream Input stream.
   */
  boost::optional<value> parse(std::istream& stream);

  /**
   * Parse a string.
   *
   * @param data String.
   */
  boost::optional<value> parse(const std::string& data);
};
}
