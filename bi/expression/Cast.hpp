/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Single.hpp"
#include "bi/common/ReturnTyped.hpp"

namespace bi {
/**
 * Cast.
 *
 * @ingroup compiler_expression
 */
class Cast: public Expression, public Single<Expression>, public ReturnTyped {
public:
  /**
   * Constructor.
   *
   * @param returnType Target type (the real return type is an optional of this type).
   * @param single Expression to cast.
   * @param loc Location.
   */
  Cast(Type* returnType, Expression* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Cast();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
