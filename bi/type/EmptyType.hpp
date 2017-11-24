/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"

namespace bi {
/**
 * Empty type.
 *
 * @ingroup birch_type
 */
class EmptyType: public Type {
public:
  /**
   * Constructor.
   *
   * @param loc Location.
   */
  EmptyType(Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~EmptyType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isEmpty() const;

  using Type::definitely;
  using Type::common;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const EmptyType& o) const;

  virtual Type* dispatchCommon(const Type& o) const;
  virtual Type* common(const EmptyType& o) const;
};
}
