/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"

namespace bi {
/**
 * C++ code generator for types.
 *
 * @ingroup compiler_io
 */
class CppTypeGenerator: public CppBaseGenerator {
public:
  CppTypeGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using indentable_ostream::visit;

  virtual void visit(const TypeParameter* o);
  virtual void visit(const TypeReference* o);
  virtual void visit(const VarDeclaration* o);
  virtual void visit(const FuncParameter* o);

protected:
  /**
   * The type being generated.
   */
  const TypeParameter* type;
};
}
