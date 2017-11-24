/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Single.hpp"
#include "bi/statement/EmptyStatement.hpp"

namespace bi {
/**
 * Statement in braces.
 *
 * @ingroup birch_expression
 */
class Braces: public Statement, public Single<Statement> {
public:
  /**
   * Constructor.
   *
   * @param single Root statement in braces.
   * @param loc Location.
   */
  Braces(Statement* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Braces();

  virtual Statement* strip();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
