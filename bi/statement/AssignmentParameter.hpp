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
class AssignmentParameter: public Statement,
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
  AssignmentParameter(shared_ptr<Name> name, Expression* single,
      Expression* braces, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~AssignmentParameter();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const AssignmentParameter& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const AssignmentParameter& o) const;
};
}
