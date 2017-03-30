/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"

namespace bi {
/**
 * C++ code generator for model view constructors.
 *
 * @ingroup compiler_io
 */
class CppViewConstructorGenerator : public CppBaseGenerator {
public:
  CppViewConstructorGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using CppBaseGenerator::visit;

  virtual void visit(const ModelParameter* o);
  virtual void visit(const VarDeclaration* o);
  virtual void visit(const FuncDeclaration* o);
};
}
