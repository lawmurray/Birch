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
class TypeOf: public Type {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  TypeOf(Location* loc = nullptr);

  virtual bool isTypeOf() const;

  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
