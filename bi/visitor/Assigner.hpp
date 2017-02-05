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

  virtual Type* modify(EmptyType* o);
  virtual Type* modify(ModelReference* o);
  virtual Type* modify(ModelParameter* o);
  virtual Type* modify(AssignableType* o);
  virtual Type* modify(BracketsType* o);
  virtual Type* modify(ParenthesesType* o);
  virtual Type* modify(TypeList* o);
};
}
