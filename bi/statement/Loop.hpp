/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/statement/Conditioned.hpp"
#include "bi/common/Braced.hpp"
#include "bi/common/Scoped.hpp"

namespace bi {
/**
 * Loop.
 *
 * @ingroup compiler_statement
 */
class Loop: public Statement, public Conditioned, public Braced, public Scoped {
public:
  /**
   * Constructor.
   *
   * @param cond Condition.
   * @param braces Body of loop.
   * @param loc Location.
   */
  Loop(Expression* cond, Expression* braces,
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Loop();

  virtual Statement* acceptClone(Cloner* visitor) const;
  virtual void acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool operator<=(Statement& o);
  virtual bool operator==(const Statement& o) const;
};
}

inline bi::Loop::~Loop() {
  //
}
