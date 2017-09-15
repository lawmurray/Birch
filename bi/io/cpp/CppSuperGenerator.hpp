/**
 * @file
 */
#pragma once

#include "bi/io/indentable_ostream.hpp"

namespace bi {
/**
 * C++ code generator for forward declarations of super types.
 *
 * @ingroup compiler_io
 */
class CppSuperGenerator: public indentable_ostream {
public:
  CppSuperGenerator(std::ostream& base, const int level = 0);

  using indentable_ostream::visit;

  virtual void visit(const Class* o);
};
}
