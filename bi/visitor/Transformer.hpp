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
 * Apply code transformations. These are typically for syntactic sugar or
 * later convenience, and are applied before any attempt to resolve symbols.
 *
 * @ingroup visitor
 */
class Transformer: public ContextualModifier {
public:
  virtual Statement* modify(Assume* o);
  virtual Statement* modify(ExpressionStatement* o);
  virtual Statement* modify(Class* o);
  virtual Statement* modify(Fiber* o);
  virtual Statement* modify(MemberFiber* o);
  virtual Statement* modify(Yield* o);
};
}
