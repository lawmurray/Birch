/**
 * @file
 */
#pragma once

#include "bi/io/indentable_ostream.hpp"

namespace bi {
/**
 * C++ code generator for forward declarations of typedefs.
 *
 * @ingroup compiler_io
 */
class CppAliasGenerator: public indentable_ostream {
public:
  CppAliasGenerator(std::ostream& base, const int level = 0);

  using indentable_ostream::visit;

  virtual void visit(const Alias* o);
};
}
