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

  virtual Statement* acceptClone(Cloner* visitor) const;
  virtual void acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual operator bool() const;

  virtual bool operator<=(Statement& o);
  virtual bool operator==(const Statement& o) const;
};
}
