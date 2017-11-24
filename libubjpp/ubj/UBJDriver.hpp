/**
 * @file
 */
#pragma once

#include "libubjpp/value.hpp"

#include <string>

class UBJDriver {
public:
  int parse(const std::string& f);
  void consume_payload();

  std::string file;
  libubjpp::object_type root;

};
