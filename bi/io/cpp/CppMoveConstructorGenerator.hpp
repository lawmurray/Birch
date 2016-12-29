/**
 * @file
 */
#pragma once

#include "bi/io/indentable_ostream.hpp"

namespace bi {
/**
 * C++ code generator for model move constructors.
 *
 * @ingroup compiler_io
 */
class CppMoveConstructorGenerator : public indentable_ostream {
public:
  CppMoveConstructorGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  virtual void visit(const ModelParameter* o);
  virtual void visit(const VarDeclaration* o);
};
}
