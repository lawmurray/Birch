/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/ReturnTyped.hpp"

namespace bi {
/**
 * Coroutine type.
 *
 * @ingroup compiler_type
 */
class CoroutineType: public Type, public ReturnTyped {
public:
  /**
   * Constructor.
   *
   * @param returnType Return type.
   * @param loc Location.
   * @param assignable Is this type assignable?
   */
  CoroutineType(Type* returnType = new EmptyType(), shared_ptr<Location> loc =
      nullptr, const bool assignable = false);

  /**
   * Destructor.
   */
  virtual ~CoroutineType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isCoroutine() const;

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const CoroutineType& o) const;
  virtual bool definitely(const ParenthesesType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const CoroutineType& o) const;
  virtual bool possibly(const ParenthesesType& o) const;
};
}
