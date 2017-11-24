/**
 * @file
 */
#pragma once

#include "libubjpp/value.hpp"

#include <string>

class JSONDriver {
public:
  libubjpp::value parse(const std::string& f);
};
