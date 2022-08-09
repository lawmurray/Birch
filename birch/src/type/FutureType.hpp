/**
 * @file
 */
#pragma once

#include "src/type/Type.hpp"
#include "src/common/Single.hpp"

namespace birch {
/**
 * Future type.
 *
 * @ingroup type
 */
class FutureType: public Type, public Single<Type> {
public:
  /**
   * Constructor.
   *
   * @param single Type.
   * @param loc Location.
   */
  FutureType(Type* single, Location* loc = nullptr);

  virtual void accept(Visitor* visitor) const;

  virtual bool isFuture() const;

  virtual Type* unwrap();
  virtual const Type* unwrap() const;
};
}
