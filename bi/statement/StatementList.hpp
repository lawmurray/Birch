/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"

namespace bi {
/**
 * Parameter list.
 *
 * @ingroup compiler_common
 */
class StatementList : public Statement {
public:
  /**
   * Constructor.
   *
   * @param head First in list.
   * @param tail Remaining list.
   * @param loc Location.
   */
  StatementList(Statement* head, Statement* tail, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~StatementList();

  /**
   * Number of objects in the list.
   */
  virtual int count() const;

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * First element of list.
   */
  Statement* head;

  /**
   * Remainder of list.
   */
  Statement* tail;
};
}
