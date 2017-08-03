/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"

namespace bi {
/**
 * C++ code generator for constructors.
 *
 * @ingroup compiler_io
 */
class CppConstructorGenerator : public CppBaseGenerator {
public:
  CppConstructorGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using CppBaseGenerator::visit;

  virtual void visit(const Class* o);
  virtual void visit(const MemberParameter* o);
  virtual void visit(const MemberVariable* o);
  virtual void visit(const MemberFunction* o);
  virtual void visit(const MemberFiber* o);
  virtual void visit(const AssignmentOperator* o);
  virtual void visit(const ConversionOperator* o);
};
}
