/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"

namespace bi {
/**
 * Empty statement.
 *
 * @ingroup compiler_statement
 */
class EmptyStatement: public Statement {
public:
  /**
   * Destructor.
   */
  virtual ~EmptyStatement();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isEmpty() const;

  virtual bool dispatchDefinitely(Statement& o);
  virtual bool definitely(EmptyStatement& o);

  virtual bool dispatchPossibly(Statement& o);
  virtual bool possibly(EmptyStatement& o);
};
}
