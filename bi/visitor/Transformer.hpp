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
 * Resolve identifiers, infer types, apply code transformations.
 *
 * @ingroup visitor
 */
class Transformer: public Modifier {
public:
  /**
   * Apply to a package.
   */
  void apply(Package* o);

  virtual Statement* modify(Assume* o);
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
