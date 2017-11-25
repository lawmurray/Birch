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
  libubjpp::value parse(const std::string& f);
};
}
