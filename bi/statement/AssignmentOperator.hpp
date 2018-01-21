/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Single.hpp"
#include "bi/common/Scoped.hpp"
#include "bi/common/Braced.hpp"

namespace bi {
/**
 * Assignment operator.
 *
 * @ingroup statement
 */
class AssignmentOperator: public Statement,
    public Named,
    public Numbered,
    public Single<Expression>,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param single Operand.
   * @param braces Body.
   * @param loc Location.
   */
  AssignmentOperator(Name* name, Expression* single,
      Statement* braces, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~AssignmentOperator();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
