/**
 * @file
 */
#pragma once

#include "src/type/Type.hpp"
#include "src/expression/Expression.hpp"
#include "src/common/Single.hpp"

namespace birch {
/**
 * Type of expression.
 *
 * @ingroup type
 */
class TypeOf: public Type, public Single<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param loc Location.
   */
  TypeOf(Expression* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~TypeOf();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
