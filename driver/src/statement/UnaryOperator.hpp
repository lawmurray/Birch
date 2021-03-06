/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/expression/Expression.hpp"
#include "src/common/Annotated.hpp"
#include "src/common/Named.hpp"
#include "src/common/Numbered.hpp"
#include "src/common/TypeParameterised.hpp"
#include "src/common/Single.hpp"
#include "src/common/ReturnTyped.hpp"
#include "src/common/Braced.hpp"

namespace birch {
/**
 * Unary operator.
 *
 * @ingroup statement
 */
class UnaryOperator: public Statement,
    public Annotated,
    public Named,
    public Numbered,
    public TypeParameterised,
    public Single<Expression>,
    public ReturnTyped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param annotation Annotation.
   * @param typeParams Generic type parameters.
   * @param name Operator name (symbol).
   * @param single Operand.
   * @param returnType Return type.
   * @param braces Body.
   * @param loc Location.
   */
  UnaryOperator(const Annotation annotation, Expression* typeParams,
      Name* name, Expression* single, Type* returnType, Statement* braces,
      Location* loc = nullptr);

  virtual void accept(Visitor* visitor) const;
};
}
