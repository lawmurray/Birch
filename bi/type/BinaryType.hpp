/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Couple.hpp"

namespace bi {
/**
 * Type of operands to a binary operator.
 *
 * @ingroup compiler_type
 */
class BinaryType: public Type, public Couple<Type> {
public:
  /**
   * Constructor.
   *
   * @param left Type of left operand.
   * @param right Type of right operand.
   * @param loc Location.
   * @param assignable Is this type assignable?
   */
  BinaryType(Type* left, Type* right, Location* loc = nullptr,
      const bool assignable = false);

  /**
   * Destructor.
   */
  virtual ~BinaryType();

  virtual bool isBinary() const;

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const BinaryType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const BinaryType& o) const;
};
}
