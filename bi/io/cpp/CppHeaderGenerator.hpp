/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"

namespace bi {
/**
 * C++ code generator for header files.
 *
 * @ingroup compiler_io
 */
class CppHeaderGenerator: public CppBaseGenerator {
public:
  CppHeaderGenerator(std::ostream& base, const int level = 0,
      const bool header = true);

  using CppBaseGenerator::visit;

  virtual void visit(const Package* o);
};
}
