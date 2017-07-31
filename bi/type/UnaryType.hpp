/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Unary.hpp"
#include "bi/common/ReturnTyped.hpp"

namespace bi {
/**
 * Unary operator type.
 *
 * @ingroup compiler_type
 */
class UnaryType: public Type, public Unary<Type>, public ReturnTyped {
public:
  /**
   * Constructor.
   *
   * @param single Operand type.
   * @param returnType Return type.
   * @param loc Location.
   * @param assignable Is this type assignable?
   */
  UnaryType(Type* single, Type* returnType = new EmptyType(),
      shared_ptr<Location> loc = nullptr, const bool assignable = false);

  /**
   * Destructor.
   */
  virtual ~UnaryType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isUnary() const;

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const UnaryType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const UnaryType& o) const;
};
}
