/**
 * @file
 */
#pragma once

#include "src/type/Type.hpp"

namespace birch {
/**
 * Empty type.
 *
 * @ingroup type
 */
class EmptyType: public Type {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  EmptyType(Location* loc = nullptr);

  virtual void accept(Visitor* visitor) const;

  virtual bool isEmpty() const;
};
}
