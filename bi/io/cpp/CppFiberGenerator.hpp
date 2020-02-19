/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"

namespace bi {
/**
 * C++ code generator for fibers.
 *
 * @ingroup io
 */
class CppFiberGenerator: public CppBaseGenerator {
public:
  CppFiberGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using CppBaseGenerator::visit;

  virtual void visit(const Fiber* o);
};
}
