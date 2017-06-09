/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"

namespace bi {
/**
 * C++ code generator for coroutines.
 *
 * @ingroup compiler_io
 */
class CppCoroutineGenerator: public CppBaseGenerator {
public:
  CppCoroutineGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using indentable_ostream::visit;

  virtual void visit(const FuncParameter* o);
  virtual void visit(const Return* o);

private:
  /**
   * Current state index.
   */
  int state;
};
}
