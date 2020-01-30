/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"

namespace bi {
/**
 * Empty type.
 *
 * @ingroup type
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
};
}
