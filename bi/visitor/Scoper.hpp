/**
 * @file
 */
#pragma once

#include "bi/visitor/ScopedModifier.hpp"

namespace bi {
/**
 * Populate scopes.
 *
 * @ingroup visitor
 */
class Scoper: public ScopedModifier {
public:
  /**
   * Constructor.
   */
  Scoper();

  /**
   * Destructor.
   */
  virtual ~Scoper();

  using ScopedModifier::modify;

  virtual Expression* modify(Parameter* o);
  virtual Statement* modify(LocalVariable* o);
  virtual Statement* modify(MemberVariable* o);
  virtual Statement* modify(GlobalVariable* o);
  virtual Statement* modify(MemberFunction* o);
  virtual Statement* modify(Function* o);
  virtual Statement* modify(MemberFiber* o);
  virtual Statement* modify(Fiber* o);
  virtual Statement* modify(BinaryOperator* o);
  virtual Statement* modify(UnaryOperator* o);
  virtual Statement* modify(Program* o);
  virtual Statement* modify(Basic* o);
  virtual Statement* modify(Class* o);
  virtual Statement* modify(Yield* o);
  virtual Expression* modify(Generic* o);
};
}
