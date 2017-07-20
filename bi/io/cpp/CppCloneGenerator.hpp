/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"

namespace bi {
/**
 * C++ code generator for @c clone() member functions.
 *
 * @ingroup compiler_io
 */
class CppCloneGenerator : public CppBaseGenerator {
public:
  CppCloneGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using CppBaseGenerator::visit;

  virtual void visit(const Class* o);
  virtual void visit(const MemberParameter* o);
  virtual void visit(const MemberVariable* o);
  virtual void visit(const MemberFunction* o);
  virtual void visit(const AssignmentOperator* o);
  virtual void visit(const ConversionOperator* o);
};
}
