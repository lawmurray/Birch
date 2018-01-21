/**
 * @file
 */
#pragma once

#include "libubjpp/value.hpp"

#include <iostream>

namespace libubjpp {
/**
 * Generator for JSON files.
 *
 * @ingroup libubjpp
 */
class JSONGenerator {
public:
  /*
   * boost::apply_visitor requirements.
   */
  using result_type = void;

  /**
   * Constructor.
   */
  JSONGenerator(std::ostream& stream);

  /**
   * Output.
   */
  void write(const value& value);

  /*
   * Output different value types.
   */
  void operator()(const object_type& value);
  void operator()(const array_type& value);
  void operator()(const string_type& value);
  void operator()(const float_type& value);
  void operator()(const double_type& value);
  void operator()(const int8_type& value);
  void operator()(const uint8_type& value);
  void operator()(const int16_type& value);
  void operator()(const int32_type& value);
  void operator()(const int64_type& value);
  void operator()(const bool_type& value);
  void operator()(const nil_type& value);
  void operator()(const noop_type& value);

private:
  /**
   * Stream.
   */
  std::ostream& stream;

  /**
   * Indent level.
   */
  int level;
};
}
