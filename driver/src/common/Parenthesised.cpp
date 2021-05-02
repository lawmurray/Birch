/**
 * @file
 */
#include "src/common/Parenthesised.hpp"

birch::Parenthesised::Parenthesised(Expression* parens) :
    parens(parens) {
  /* pre-condition */
  assert(parens);
}
