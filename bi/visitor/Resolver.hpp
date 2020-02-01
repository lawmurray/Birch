/**
 * @file
 */
#pragma once

#include "bi/visitor/ScopedModifier.hpp"

namespace bi {
/**
 * Resolve identifiers.
 *
 * @ingroup visitor
 */
class Resolver: public ScopedModifier {
public:
  /**
   * Constructor.
   */
  Resolver();

  /**
   * Destructor.
   */
  virtual ~Resolver();

  using ScopedModifier::modify;

  virtual Expression* modify(NamedExpression* o);
  virtual Type* modify(NamedType* o);
};
}
