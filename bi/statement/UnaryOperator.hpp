/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/expression/Expression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Unary.hpp"
#include "bi/common/ReturnTyped.hpp"
#include "bi/common/Scoped.hpp"
#include "bi/common/Braced.hpp"

namespace bi {
/**
 * Unary operator.
 *
 * @ingroup compiler_statement
 */
class UnaryOperator: public Statement,
    public Named,
    public Numbered,
    public Unary<Expression>,
    public ReturnTyped,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param single Operand.
   * @param returnType Return type.
   * @param braces Braces expression.
   * @param loc Location.
   */
  UnaryOperator(shared_ptr<Name> name, Expression* single, Type* returnType,
      Statement* braces, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~UnaryOperator();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const UnaryOperator& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const UnaryOperator& o) const;
};
}
