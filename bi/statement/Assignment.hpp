/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/statement/AssignmentOperator.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Binary.hpp"
#include "bi/common/Reference.hpp"

namespace bi {
/**
 * Reference to assignment operator.
 *
 * @ingroup compiler_statement
 */
class Assignment: public Statement, public Named, public Binary<
    Expression>, public Reference<AssignmentOperator> {
public:
  /**
   * Constructor.
   *
   * @param left Left operand.
   * @param name Name.
   * @param right Right operand.
   * @param loc Location.
   * @param target Target.
   */
  Assignment(Expression* left, shared_ptr<Name> name,
      Expression* right, shared_ptr<Location> loc = nullptr,
      AssignmentOperator* target = nullptr);

  /**
   * Destructor.
   */
  virtual ~Assignment();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const Assignment& o) const;
  virtual bool definitely(const AssignmentOperator& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const Assignment& o) const;
  virtual bool possibly(const AssignmentOperator& o) const;
};
}
