/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Numbered.hpp"
#include "src/common/Single.hpp"
#include "src/common/Scoped.hpp"
#include "src/common/Braced.hpp"

namespace birch {
/**
 * Assignment operator.
 *
 * @ingroup statement
 */
class AssignmentOperator: public Statement, public Numbered, public Single<
    Expression>, public Scoped, public Braced {
public:
  /**
   * Constructor.
   *
   * @param single Operand.
   * @param braces Body.
   * @param loc Location.
   */
  AssignmentOperator(Expression* single, Statement* braces, Location* loc =
      nullptr);

  /**
   * Destructor.
   */
  virtual ~AssignmentOperator();

  virtual bool isDeclaration() const;

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
