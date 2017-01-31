/**
 * @file
 */
#include "bi/common/Formed.hpp"

#include "bi/common/List.hpp"

bi::Formed::Formed(Expression* parens, const FunctionForm form) : Parenthesised(parens), form(form) {
  //
}

bi::Formed::~Formed() {
  //
}

bool bi::Formed::isBinary() const {
  return form == BINARY_OPERATOR || form == ASSIGNMENT_OPERATOR;
}

bool bi::Formed::isUnary() const {
  return form == UNARY_OPERATOR;
}

bool bi::Formed::isAssignment() const {
  return form == ASSIGNMENT_OPERATOR;
}

bool bi::Formed::isConstructor() const {
  return form == CONSTRUCTOR;
}

const bi::Expression* bi::Formed::getLeft() const {
  /* pre-condition */
  assert(isBinary());

  ExpressionList* expr = dynamic_cast<ExpressionList*>(parens->strip());
  assert(expr);
  return expr->head.get();
}

const bi::Expression* bi::Formed::getRight() const {
  /* pre-condition */
  assert(isBinary() || isUnary());

  if (isBinary()) {
    ExpressionList* expr = dynamic_cast<ExpressionList*>(parens->strip());
    assert(expr);
    return expr->tail.get();
  } else {
    assert(isUnary());
    return parens->strip();
  }
}
