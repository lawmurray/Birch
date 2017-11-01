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
class AnyType: public Type {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  AnyType(Location* loc);

  /**
   * Destructor.
   */
  virtual ~AnyType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isEmpty() const;

  using Type::definitely;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const AnyType& o) const;
  virtual bool definitely(const GenericType& o) const;
};
}
