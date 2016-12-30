/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"

namespace bi {
/**
 * Empty expression.
 *
 * @ingroup compiler_expression
 */
class EmptyExpression: public Expression {
public:
  /**
   * Constructor.
   */
  EmptyExpression();

  /**
   * Destructor.
   */
  virtual ~EmptyExpression();

  virtual Expression* acceptClone(Cloner* visitor) const;
  virtual void acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual operator bool() const;

  virtual bool operator<=(Expression& o);
  virtual bool operator==(const Expression& o) const;
};
}
