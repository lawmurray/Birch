/**
 * @file
 */
#pragma once

#include "bi/io/indentable_ostream.hpp"

namespace bi {
/**
 * C++ code generator for forward declarations of types.
 *
 * @ingroup compiler_io
 */
class CppForwardGenerator: public indentable_ostream {
public:
  CppForwardGenerator(std::ostream& base, const int level = 0);

  using indentable_ostream::visit;

  virtual void visit(const Class* o);
  virtual void visit(const Alias* o);

private:
  /**
   * Classes that have already been forward declared.
   */
  std::set<const Class*> classes;

  /**
   * Alias types that have already been forward declared.
   */
  std::set<const Alias*> aliases;
};
}
