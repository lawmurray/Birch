/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/expression/Expression.hpp"
#include "src/common/Annotated.hpp"
#include "src/common/Braced.hpp"

namespace birch {
/**
 * For loop.
 *
 * @ingroup statement
 */
class For: public Statement, public Annotated, public Braced {
public:
  /**
   * Constructor.
   *
   * @param annotation Annotation.
   * @param index Index.
   * @param from From expression.
   * @param to To expression.
   * @param braces Body of loop.
   * @param loc Location.
   */
  For(const Annotation annotation, Statement* index, Expression* from,
      Expression* to, Statement* braces, Location* loc = nullptr);

  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Index.
   */
  Statement* index;

  /**
   * From expression.
   */
  Expression* from;

  /**
   * To expression.
   */
  Expression* to;
};
}
