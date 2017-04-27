/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"

namespace bi {
/**
 * C++ code generator for constructors.
 *
 * @ingroup compiler_io
 */
class CppConstructorGenerator : public CppBaseGenerator {
public:
  CppConstructorGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using CppBaseGenerator::visit;

  virtual void visit(const TypeParameter* o);
  virtual void visit(const VarParameter* o);
  virtual void visit(const VarDeclaration* o);
  virtual void visit(const FuncDeclaration* o);
  virtual void visit(const ConversionDeclaration* o);

private:
  bool inInitial;
};
}
