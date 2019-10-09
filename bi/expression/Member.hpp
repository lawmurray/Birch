/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Couple.hpp"

namespace bi {
/**
 * Membership operator expression.
 *
 * @ingroup expression
 */
class Member: public Expression, public Couple<Expression> {
public:
  /**
   * Constructor.
   *
   * @param left Left operand.
   * @param right Right operand.
   * @param loc Location.
   */
  Member(Expression* left, Expression* right, Location* loc =
      nullptr);

  /**
   * Destructor.
   */
  virtual ~Member();

  virtual bool isAssignable() const;

  virtual Lookup lookup(Expression* args);
  virtual MemberVariable* resolve(Call<MemberVariable>* o);
  virtual MemberFunction* resolve(Call<MemberFunction>* o);
  virtual MemberFiber* resolve(Call<MemberFiber>* o);

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
