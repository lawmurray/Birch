/**
 * @file
 */
#pragma once

#include "bi/visitor/Modifier.hpp"

namespace bi {
/**
 * Visitor to make all types assignable.
 *
 * @ingroup compiler_visitor
 */
class Assigner : public Modifier {
public:
  /**
   * Destructor.
   */
  virtual ~Assigner();

  using Modifier::modify;

  virtual Type* modify(EmptyType* o);
  virtual Type* modify(BracketsType* o);
  virtual Type* modify(ParenthesesType* o);
  virtual Type* modify(FunctionType* o);
  virtual Type* modify(CoroutineType* o);
  virtual Type* modify(List<Type>* o);
  virtual Type* modify(IdentifierType<Class>* o);
  virtual Type* modify(IdentifierType<AliasType>* o);
  virtual Type* modify(IdentifierType<BasicType>* o);
};
}
