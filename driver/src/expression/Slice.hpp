/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/common/Bracketed.hpp"
#include "src/common/Single.hpp"

namespace birch {
/**
 * Expression with proceeding square brackets.
 *
 * @ingroup expression
 */
class Slice: public Expression, public Single<Expression>, public Bracketed {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param brackets Brackets.
   * @param loc Location.
   */
  Slice(Expression* single, Expression* brackets, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Slice();

  virtual bool isAssignable() const;

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isSlice() const;
};
}
