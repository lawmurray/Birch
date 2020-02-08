/**
 * @file
 */
#pragma once

#include "bi/visitor/Modifier.hpp"
#include "bi/visitor/Cloner.hpp"
#include "bi/expression/all.hpp"
#include "bi/statement/all.hpp"
#include "bi/type/all.hpp"

namespace bi {
/**
 * Apply code transformations.
 *
 * @ingroup visitor
 */
class Transformer: public Modifier {
public:
  virtual Statement* modify(Assume* o);
  virtual Statement* modify(Class* o);
  virtual Statement* modify(Fiber* o);
  virtual Statement* modify(MemberFiber* o);
  virtual Statement* modify(ExpressionStatement* o);

private:
  /*
   * Auxiliary visitors.
   */
  Cloner cloner;
};
}
