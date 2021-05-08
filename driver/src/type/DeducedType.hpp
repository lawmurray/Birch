/**
 * @file
 */
#pragma once

#include "src/type/Type.hpp"

namespace birch {
/**
 * Deduced return type.
 *
 * @ingroup type
 */
class DeducedType: public Type {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  DeducedType(Location* loc = nullptr);

  virtual bool isDeduced() const;

  virtual void accept(Visitor* visitor) const;
};
}
