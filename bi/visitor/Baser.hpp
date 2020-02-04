/**
 * @file
 */
#pragma once

#include "bi/visitor/ScopedModifier.hpp"

namespace bi {
/**
 * Resolve inheritance relationships between classes and link up scopes
 * accordingly.
 *
 * @ingroup visitor
 */
class Baser: public ScopedModifier {
public:
  /**
   * Constructor.
   */
  Baser();

  /**
   * Destructor.
   */
  virtual ~Baser();

  using ScopedModifier::modify;

  virtual Statement* modify(Class* o);
};
}
