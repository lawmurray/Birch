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
  CppPackageGenerator(std::ostream& base, const int level, const bool header,
      const bool includeInline, const bool includeLines);

  using CppGenerator::visit;

  virtual void visit(const Package* o);
};
}
