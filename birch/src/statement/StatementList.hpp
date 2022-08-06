/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"

namespace birch {
/**
 * Parameter list.
 *
 * @ingroup common
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

  virtual int count() const;
  virtual bool isEmpty() const;

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
