/**
 * @file
 */
#pragma once

#include "bi/io/indentable_ostream.hpp"

namespace bi {
/**
 * C++ code generator for model constructors.
 *
 * @ingroup compiler_io
 */
class CppConstructorGenerator : public indentable_ostream {
public:
  CppConstructorGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using indentable_ostream::visit;

  virtual void visit(const BracketsType* o);
  virtual void visit(const ModelReference* o);
  virtual void visit(const ModelParameter* o);

  void initialise(const VarParameter* o);
  void assign(const VarParameter* o);
};
}
