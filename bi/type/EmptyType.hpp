/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"

namespace bi {
/**
 * Empty type.
 *
 * @ingroup compiler_type
 */
class EmptyType: public Type {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   * @param assignable Is this type writeable?
   */
  EmptyType(shared_ptr<Location> loc = nullptr, const bool assignable = false);

  /**
   * Destructor.
   */
  virtual ~EmptyType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isEmpty() const;

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(Type& o);
  virtual bool definitely(EmptyType& o);

  virtual bool dispatchPossibly(Type& o);
  virtual bool possibly(EmptyType& o);
};
}
