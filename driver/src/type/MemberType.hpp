/**
 * @file
 */
#pragma once

#include "src/type/Type.hpp"
#include "src/common/Couple.hpp"

namespace birch {
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

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual int depth() const;
  virtual bool isMembership() const;
  virtual bool isValue() const;
};
}
