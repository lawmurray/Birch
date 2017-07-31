/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Binary.hpp"
#include "bi/common/ReturnTyped.hpp"

namespace bi {
/**
 * Function type.
 *
 * @ingroup compiler_type
 */
class BinaryType: public Type, public Binary<Type>, public ReturnTyped {
public:
  /**
   * Constructor.
   *
   * @param left Left operand type.
   * @param right Right operand type.
   * @param returnType Return type.
   * @param loc Location.
   * @param assignable Is this type assignable?
   */
  BinaryType(Type* left, Type* right, Type* returnType = new EmptyType(),
      shared_ptr<Location> loc = nullptr, const bool assignable = false);

  /**
   * Destructor.
   */
  virtual ~BinaryType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isBinary() const;

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const BinaryType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const BinaryType& o) const;
};
}
