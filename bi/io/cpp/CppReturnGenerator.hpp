/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"

namespace bi {
/**
 * C++ code generator for return statements.
 *
 * @ingroup compiler_io
 */
class CppReturnGenerator: public CppBaseGenerator {
public:
  CppReturnGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using CppBaseGenerator::visit;

  virtual void visit(const VarParameter* o);
  virtual void visit(const FuncParameter* o);
};
}
