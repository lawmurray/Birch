/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"

namespace bi {
/**
 * C++ code generator for model assignment operators.
 *
 * @ingroup compiler_io
 */
class CppAssignmentGenerator : public CppBaseGenerator {
public:
  CppAssignmentGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using CppBaseGenerator::visit;
  virtual void visit(const ModelParameter* o);
  virtual void visit(const VarParameter* o);
  virtual void visit(const FuncParameter* o);
};
}
