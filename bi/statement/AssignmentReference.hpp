/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/statement/AssignmentParameter.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Binary.hpp"
#include "bi/common/Reference.hpp"

namespace bi {
/**
 * Reference to assignment operator.
 *
 * @ingroup compiler_expression
 */
class AssignmentReference: public Statement, public Named, public Binary<
    Expression>, public Reference<AssignmentParameter> {
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
  AssignmentReference(Expression* left, shared_ptr<Name> name,
      Expression* right, shared_ptr<Location> loc = nullptr,
      const AssignmentParameter* target = nullptr);

  /**
   * Destructor.
   */
  virtual ~AssignmentReference();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const AssignmentReference& o) const;
  virtual bool definitely(const AssignmentParameter& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const AssignmentReference& o) const;
  virtual bool possibly(const AssignmentParameter& o) const;
};
}
