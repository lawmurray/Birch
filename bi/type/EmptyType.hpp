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
   */
  EmptyType(shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~EmptyType();

  virtual Type* acceptClone(Cloner* visitor) const;
  virtual void acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual operator bool() const;

  virtual bool operator<=(Type& o);
  virtual bool operator==(const Type& o) const;
};
}
