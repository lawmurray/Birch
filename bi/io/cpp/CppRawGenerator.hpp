/**
 * @file
 */
#pragma once

#include "bi/io/indentable_ostream.hpp"

namespace bi {
/**
 * C++ code generator for raw code at global scope.
 *
 * @ingroup birch_io
 */
class CppRawGenerator: public indentable_ostream {
public:
  CppRawGenerator(std::ostream& base, const int level = 0, const bool header =
      false);

  using indentable_ostream::visit;

  virtual void visit(const Function* o);
  virtual void visit(const Fiber* o);
  virtual void visit(const Program* o);
  virtual void visit(const Class* o);
  virtual void visit(const Raw* o);

protected:
  /**
   * Output header instead of source?
   */
  bool header;
};
}
