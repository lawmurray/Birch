/**
 * @file
 */
#pragma once

#include "src/generate/CppGenerator.hpp"

namespace birch {
/**
 * C++ code generator for header files.
 *
 * @ingroup io
 */
class CppPackageGenerator: public CppGenerator {
public:
  CppPackageGenerator(std::ostream& base, const int level = 0,
      const bool header = true);

  using CppGenerator::visit;

  virtual void visit(const Package* o);
};
}
