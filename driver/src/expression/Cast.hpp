/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/common/Single.hpp"
#include "src/common/ReturnTyped.hpp"

namespace birch {
/**
 * Cast.
 *
 * @ingroup expression
 */
class Cast: public Expression, public Single<Expression>, public ReturnTyped {
public:
  /**
   * Constructor.
   *
   * @param returnType Target type (the real return type is an optional of
   * this type).
   * @param single Expression to cast.
   * @param loc Location.
   */
  Cast(Type* returnType, Expression* single, Location* loc = nullptr);

  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
