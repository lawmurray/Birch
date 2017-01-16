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
class CppDispatchGenerator: public CppBaseGenerator {
public:
  CppDispatchGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  virtual void visit(const FuncParameter* o);
};
}
