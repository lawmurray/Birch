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

  /**
   * Destructor.
   */
  virtual ~TypeOf();

  virtual bool isTypeOf() const;

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
