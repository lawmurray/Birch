/**
 * @file
 */
#pragma once

#include "bi/io/cpp/CppBaseGenerator.hpp"

namespace bi {
/**
 * C++ code generator for member coroutines.
 *
 * @ingroup compiler_io
 */
class CppMemberCoroutineGenerator: public CppBaseGenerator {
public:
  CppMemberCoroutineGenerator(const Class* type, std::ostream& base,
      const int level = 0, const bool header = false);

  using indentable_ostream::visit;

  virtual void visit(const MemberCoroutine* o);
  virtual void visit(const Return* o);
  virtual void visit(const Identifier<LocalVariable>* o);
  virtual void visit(const Identifier<MemberVariable>* o);
  virtual void visit(const LocalVariable* o);

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
