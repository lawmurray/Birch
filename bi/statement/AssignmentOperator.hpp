/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Unary.hpp"
#include "bi/common/Scoped.hpp"
#include "bi/common/Braced.hpp"

namespace bi {
/**
 * Assignment operator.
 *
 * @ingroup compiler_expression
 */
class AssignmentOperator: public Statement,
    public Named,
    public Numbered,
    public Unary<Expression>,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param single Operand.
   * @param braces Braces expression.
   * @param loc Location.
   */
  AssignmentOperator(shared_ptr<Name> name, Expression* single,
      Statement* braces, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~AssignmentOperator();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const AssignmentOperator& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const AssignmentOperator& o) const;
};
}
