/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/expression/Expression.hpp"
#include "bi/common/Annotated.hpp"
#include "bi/common/Braced.hpp"
#include "bi/common/Scoped.hpp"

namespace bi {
/**
 * For loop.
 *
 * @ingroup statement
 */
class For: public Statement, public Annotated, public Scoped, public Braced {
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
  For(const Annotation annotation, Expression* index, Expression* from,
      Expression* to, Statement* braces, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~For();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Index.
   */
  Expression* index;

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
