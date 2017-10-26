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
   * Constructor.
   *
   * @param loc Location.
   */
  EmptyStatement(Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~EmptyStatement();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isEmpty() const;
};
}
