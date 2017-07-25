/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppCoroutineGenerator.hpp"

namespace bi {
/**
 * C++ code generator for member coroutines.
 *
 * @ingroup compiler_io
 */
class CppMemberCoroutineGenerator: public CppCoroutineGenerator {
public:
  CppMemberCoroutineGenerator(const Class* type, std::ostream& base,
      const int level = 0, const bool header = false);

  using indentable_ostream::visit;

  virtual void visit(const MemberCoroutine* o);
  virtual void visit(const Identifier<MemberParameter>* o);
  virtual void visit(const Identifier<MemberVariable>* o);

private:
  /**
   * The class being generated.
   */
  const Class* type;

  /**
   * Current state index.
   */
  int state;
};
}
