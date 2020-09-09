/**
 * @file
 */
#pragma once

#include "src/generate/CppGenerator.hpp"

namespace birch {
/**
 * C++ code generator for types.
 *
 * @ingroup io
 */
class CppClassGenerator: public CppGenerator {
public:
  CppClassGenerator(std::ostream& base, const int level = 0,
      const bool header = false, const bool generic = false,
      const Class* currentClass = nullptr);

  using CppGenerator::visit;

  virtual void visit(const Class* o);
  virtual void visit(const MemberVariable* o);
  virtual void visit(const MemberFunction* o);
  virtual void visit(const AssignmentOperator* o);
  virtual void visit(const ConversionOperator* o);

protected:
  /**
   * The class being generated.
   */
  const Class* currentClass;

  /**
   * Generate code for the base type.
   */
  void genBase(const Class* o);
};
}
