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
#include "src/common/Couple.hpp"
#include "src/common/ReturnTyped.hpp"
#include "src/common/Braced.hpp"

namespace birch {
/**
 * Binary operator.
 *
 * @ingroup statement
 */
class BinaryOperator: public Statement,
    public Annotated,
    public Named,
    public Numbered,
    public TypeParameterised,
    public Couple<Expression>,
    public ReturnTyped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param annotation Annotation.
   * @param typeParams Generic type parameters.
   * @param left Left operand.
   * @param name Operator name (symbol).
   * @param right Right operand.
   * @param returnType Return type.
   * @param braces Body.
   * @param loc Location.
   */
  BinaryOperator(const Annotation annotation, Expression* typeParams,
      Expression* left, Name* name, Expression* right, Type* returnType,
      Statement* braces, Location* loc = nullptr);

  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
