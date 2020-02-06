/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppFiberGenerator.hpp"

namespace bi {
/**
 * C++ code generator for member Fibers.
 *
 * @ingroup io
 */
class CppMemberFiberGenerator: public CppFiberGenerator {
public:
  CppMemberFiberGenerator(const Class* type, std::ostream& base,
      const int level = 0, const bool header = false);

  using CppFiberGenerator::visit;

  virtual void visit(const MemberFiber* o);

private:
  /**
   * The class to which the member fiber belongs.
   */
  const Class* theClass;
};
}
