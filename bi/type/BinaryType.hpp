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
 * @ingroup type
 */
class BinaryType: public Type, public Couple<Type> {
public:
  /**
   * Constructor.
   *
   * @param left Type of left operand.
   * @param right Type of right operand.
   * @param loc Location.
   */
  BinaryType(Type* left, Type* right, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~BinaryType();

  virtual bool isBinary() const;

  virtual Type* getLeft() const;
  virtual Type* getRight() const;

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::isConvertible;
  using Type::isAssignable;

  virtual bool dispatchIsConvertible(const Type& o) const;
  virtual bool isConvertible(const BinaryType& o) const;

  virtual bool dispatchIsAssignable(const Type& o) const;
  virtual bool isAssignable(const BinaryType& o) const;
};
}
