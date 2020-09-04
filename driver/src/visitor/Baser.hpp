/**
 * @file
 */
#pragma once

#include "src/visitor/ScopedModifier.hpp"

namespace birch {
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
