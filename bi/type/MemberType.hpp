/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Couple.hpp"

namespace bi {
/**
 * Membership operator type expression.
 *
 * @ingroup expression
 */
class MemberType: public Type, public Couple<Type> {
public:
  /**
   * Constructor.
   *
   * @param left Left operand.
   * @param right Right operand.
   * @param loc Location.
   */
  MemberType(Type* left, Type* right, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~MemberType();

  virtual int depth() const;

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
