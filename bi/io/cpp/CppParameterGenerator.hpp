/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"

namespace bi {
/**
 * C++ code generator for formal parameters.
 *
 * @ingroup compiler_io
 */
class CppParameterGenerator: public CppBaseGenerator {
public:
  CppParameterGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using CppBaseGenerator::visit;

  virtual void visit(const TypeReference* o);
  virtual void visit(const BracketsType* o);
  virtual void visit(const ParenthesesType* o);
  virtual void visit(const DelayType* o);
  virtual void visit(const LambdaType* o);

  virtual void visit(const VarParameter* o);
  virtual void visit(const FuncParameter* o);
};
}
