/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"

namespace bi {
/**
 * C++ code generator for formal template parameters.
 *
 * @ingroup compiler_io
 */
class CppTemplateParameterGenerator: public CppBaseGenerator {
public:
  CppTemplateParameterGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using CppBaseGenerator::visit;

  virtual void visit(const VarParameter* o);
  virtual void visit(const FuncParameter* o);
  virtual void visit(const Dispatcher* o);

private:
  /**
   * Cumulative count of assignable arguments.
   */
  int args;
};
}
