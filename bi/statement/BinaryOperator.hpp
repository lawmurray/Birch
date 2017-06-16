/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/expression/Expression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Binary.hpp"
#include "bi/common/ReturnTyped.hpp"
#include "bi/common/Scoped.hpp"
#include "bi/common/Braced.hpp"

namespace bi {
/**
 * Binary operator.
 *
 * @ingroup compiler_expression
 */
class BinaryOperator: public Statement,
    public Named,
    public Numbered,
    public Binary<Expression>,
    public ReturnTyped,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param left Left operand.
   * @param name Name.
   * @param right Right operand.
   * @param returnType Return type.
   * @param braces Braces expression.
   * @param loc Location.
   */
  BinaryOperator(Expression* left, shared_ptr<Name> name, Expression* right,
      Type* returnType, Statement* braces, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~BinaryOperator();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const BinaryOperator& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const BinaryOperator& o) const;
};
}
