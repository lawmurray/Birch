/**
 * @file
 */
#pragma once

#include "src/generate/CppGenerator.hpp"

namespace birch {
/**
 * C++ code generator for structs.
 *
 * @ingroup io
 */
class CppStructGenerator: public CppGenerator {
public:
  CppStructGenerator(std::ostream& base, const int level, const bool header,
      const bool includeInline, const bool includeLines,
      const Struct* currentStruct);

  using CppGenerator::visit;

  virtual void visit(const Struct* o);
  virtual void visit(const MemberVariable* o);

protected:
  /**
   * The struct being generated.
   */
  const Struct* currentStruct;

  /**
   * Generate code for the base type of a struct.
   * 
   * @param o The struct.
   */
  void genBase(const Struct* o);
};
}
