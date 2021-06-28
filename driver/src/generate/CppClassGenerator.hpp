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
  CppClassGenerator(std::ostream& base, const int level, const bool header,
      const bool includeInline, const bool includeLines,
      const Class* currentClass);

  using CppGenerator::visit;

  virtual void visit(const Class* o);
  virtual void visit(const MemberVariable* o);
  virtual void visit(const MemberFunction* o);
  virtual void visit(const AssignmentOperator* o);
  virtual void visit(const ConversionOperator* o);
  virtual void visit(const SliceOperator* o);

protected:
  /**
   * The class being generated.
   */
  const Class* currentClass;

  /**
   * Generate code for the base type of a class.
   * 
   * @param o The class.
   * @param includeTypename Should the typename be included (if determined
   * necessary)?
   * 
   */
  void genBase(const Class* o, const bool includeTypename);
};
}
