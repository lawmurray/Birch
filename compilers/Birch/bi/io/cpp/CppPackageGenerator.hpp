/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"

namespace bi {
/**
 * C++ code generator for header files.
 *
 * @ingroup io
 */
class CppPackageGenerator: public CppBaseGenerator {
public:
  CppPackageGenerator(std::ostream& base, const int level = 0,
      const bool header = true, const bool absolute = false);

  using CppBaseGenerator::visit;

  virtual void visit(const Package* o);
};
}
