/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/expression/Expression.hpp"
#include "src/common/Annotated.hpp"
#include "src/common/Named.hpp"
#include "src/common/Numbered.hpp"
#include "src/common/Couple.hpp"
#include "src/common/ReturnTyped.hpp"
#include "src/common/Scoped.hpp"
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
    public Couple<Expression>,
    public ReturnTyped,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param annotation Annotation.
   * @param left Left operand.
   * @param name Operator name (symbol).
   * @param right Right operand.
   * @param returnType Return type.
   * @param braces Body.
   * @param loc Location.
   */
  BinaryOperator(const Annotation annotation, Expression* left, Name* name,
      Expression* right, Type* returnType, Statement* braces,
      Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~BinaryOperator();

  virtual bool isDeclaration() const;

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
