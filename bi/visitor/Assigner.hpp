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
  virtual Type* modify(ModelReference* o);
  virtual Type* modify(ModelParameter* o);
  virtual Type* modify(BracketsType* o);
  virtual Type* modify(ParenthesesType* o);
  virtual Type* modify(DelayType* o);
  virtual Type* modify(LambdaType* o);
  virtual Type* modify(TypeList* o);
  virtual Type* modify(VariantType* o);
};
}
