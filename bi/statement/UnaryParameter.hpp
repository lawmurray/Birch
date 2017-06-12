/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/expression/Expression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Unary.hpp"
#include "bi/common/Typed.hpp"
#include "bi/common/Scoped.hpp"
#include "bi/common/Braced.hpp"

namespace bi {
/**
 * Unary operator.
 *
 * @ingroup compiler_expression
 */
class UnaryParameter: public Statement,
    public Named,
    public Numbered,
    public Unary<Expression>,
    public Typed,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param single Operand.
   * @param type Typed type.
   * @param braces Braces expression.
   * @param loc Location.
   */
  UnaryParameter(shared_ptr<Name> name, Expression* single, Type* type,
      Expression* braces, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~UnaryParameter();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const UnaryParameter& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const UnaryParameter& o) const;
};
}
