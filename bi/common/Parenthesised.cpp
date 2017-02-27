/**
 * @file
 */
#include "bi/common/Parenthesised.hpp"

#include "bi/common/List.hpp"

bi::Parenthesised::Parenthesised(Expression* parens) :
    parens(parens) {
  /* pre-condition */
  assert(parens);
}

bi::Parenthesised::~Parenthesised() {
  //
}

const bi::Expression* bi::Parenthesised::getLeft() const {
  Expression* strip = parens->strip();
  ExpressionList* expr = dynamic_cast<ExpressionList*>(strip);
  if (expr) {
    return expr->head.get();
  } else {
    return strip;
  }
}

const bi::Expression* bi::Parenthesised::getRight() const {
  Expression* strip = parens->strip();
  ExpressionList* expr = dynamic_cast<ExpressionList*>(strip);
  if (expr) {
    return expr->tail.get();
  } else {
    return strip;
  }
}
