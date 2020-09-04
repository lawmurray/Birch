/**
 * @file
 */
#pragma once

#include "src/type/Type.hpp"
#include "src/common/Single.hpp"

namespace birch {
/**
 * Optional type.
 *
 * @ingroup type
 */
class OptionalType: public Type, public Single<Type> {
public:
  /**
   * Constructor.
   *
   * @param single Type.
   * @param loc Location.
   */
  OptionalType(Type* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~OptionalType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isOptional() const;
  virtual bool isValue() const;

  virtual Type* unwrap();
  virtual const Type* unwrap() const;
};
}
