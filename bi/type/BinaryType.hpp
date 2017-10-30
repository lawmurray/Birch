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

  using Type::definitely;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const BinaryType& o) const;
};
}
