/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/expression/Expression.hpp"
#include "bi/common/Annotated.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Single.hpp"
#include "bi/common/ReturnTyped.hpp"
#include "bi/common/Scoped.hpp"
#include "bi/common/Braced.hpp"

namespace bi {
/**
 * Unary operator.
 *
 * @ingroup statement
 */
class UnaryOperator: public Statement,
    public Annotated,
    public Named,
    public Numbered,
    public Single<Expression>,
    public ReturnTyped,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param annotation Annotation.
   * @param name Operator name (symbol).
   * @param single Operand.
   * @param returnType Return type.
   * @param braces Body.
   * @param loc Location.
   */
  UnaryOperator(const Annotation annotation, Name* name, Expression* single,
      Type* returnType, Statement* braces, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~UnaryOperator();

  virtual bool isDeclaration() const;

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
