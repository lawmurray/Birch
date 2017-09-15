/**
 * @file
 */
#pragma once

#include "bi/io/indentable_ostream.hpp"

namespace bi {
/**
 * C++ code generator for header files.
 *
 * @ingroup compiler_io
 */
class CppHeaderGenerator: public indentable_ostream {
public:
  CppHeaderGenerator(std::ostream& base, const int level = 0);

  using indentable_ostream::visit;

  virtual void visit(const Package* o);
};
}
