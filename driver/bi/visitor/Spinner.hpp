/**
 * @file
 */
#pragma once

#include "bi/visitor/ContextualModifier.hpp"
#include "bi/expression/all.hpp"
#include "bi/statement/all.hpp"
#include "bi/type/all.hpp"

namespace bi {
/**
 * Unwind use of the spin operator into a preceding while loop that yields
 * until the fiber is complete.
 *
 * @ingroup visitor
 */
class Spinner: public ContextualModifier {
public:
  virtual Statement* modify(ExpressionStatement* o);
  virtual Statement* modify(Assume* o);
  virtual Statement* modify(LocalVariable* o);
  virtual Statement* modify(If* o);
  virtual Statement* modify(For* o);
  virtual Statement* modify(Parallel* o);
  virtual Statement* modify(While* o);
  virtual Statement* modify(DoWhile* o);
  virtual Statement* modify(Assert* o);
  virtual Statement* modify(Return* o);
  virtual Statement* modify(Yield* o);

protected:
  /**
   * For a given expression, construct loop statements associated with any
   * spin operations, to precede the expression.
   *
   * @param o The expression.
   * @param loops Existing loops to be added in the same place.
   *
   * @return Loop statements to precede the expression; `nullptr` if none.
   */
  Statement* extract(Expression* o, Statement* loops = nullptr);
};
}
