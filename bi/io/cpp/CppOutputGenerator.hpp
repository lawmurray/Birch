/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"

namespace bi {
/**
 * C++ code generator for output variable declarations.
 *
 * @ingroup compiler_io
 */
class CppOutputGenerator: public CppBaseGenerator {
public:
  CppOutputGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  virtual void visit(const FuncParameter* o);
};
}
