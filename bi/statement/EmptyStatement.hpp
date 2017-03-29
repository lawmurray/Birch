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

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const EmptyStatement& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const EmptyStatement& o) const;
};
}
