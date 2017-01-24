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

  virtual void visit(const BracketsType* o);
  virtual void visit(const ModelReference* o);
  virtual void visit(const ModelParameter* o);

  void initialise(const VarDeclaration* o);
  void assign(const VarDeclaration* o);
};
}
