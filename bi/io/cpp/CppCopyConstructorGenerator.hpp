/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"

namespace bi {
/**
 * C++ code generator for copy constructors.
 *
 * @ingroup compiler_io
 */
class CppCopyConstructorGenerator : public CppBaseGenerator {
public:
  CppCopyConstructorGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using CppBaseGenerator::visit;

  virtual void visit(const TypeParameter* o);
  virtual void visit(const VarDeclaration* o);
  virtual void visit(const FuncDeclaration* o);

private:
  bool before;
};
}
