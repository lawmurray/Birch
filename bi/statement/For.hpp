/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Braced.hpp"
#include "bi/common/Scoped.hpp"

namespace bi {
/**
 * For.
 *
 * @ingroup compiler_statement
 */
class For: public Statement, public Braced, public Scoped {
public:
  /**
   * Constructor.
   *
   * @param index Index.
   * @param from From expression.
   * @param to To expression.
   * @param braces Body of loop.
   * @param loc Location.
   */
  For(Expression* index, Expression* from, Expression* to, Expression* braces,
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~For();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const For& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const For& o) const;

  /**
   * Index.
   */
  unique_ptr<Expression> index;

  /**
   * From expression.
   */
  unique_ptr<Expression> from;

  /**
   * To expression.
   */
  unique_ptr<Expression> to;
};
}
