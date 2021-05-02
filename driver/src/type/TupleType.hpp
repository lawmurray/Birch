/**
 * @file
 */
#pragma once

#include "src/type/Type.hpp"
#include "src/common/Single.hpp"

namespace birch {
/**
 * Tuple type.
 *
 * @ingroup type
 */
class TupleType: public Type, public Single<Type> {
public:
  /**
   * Constructor.
   *
   * @param single Type in parentheses.
   * @param loc Location.
   */
  TupleType(Type* single, Location* loc = nullptr);

  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isTuple() const;
  virtual bool isValue() const;
};
}
