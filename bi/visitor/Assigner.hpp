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
  virtual Type* modify(ArrayType* o);
  virtual Type* modify(TupleType* o);
  virtual Type* modify(BinaryType* o);
  virtual Type* modify(FunctionType* o);
  virtual Type* modify(OverloadedType* o);
  virtual Type* modify(FiberType* o);
  virtual Type* modify(OptionalType* o);
  virtual Type* modify(TypeList* o);
  virtual Type* modify(BasicType* o);
  virtual Type* modify(ClassType* o);
  virtual Type* modify(AliasType* o);
};
}
