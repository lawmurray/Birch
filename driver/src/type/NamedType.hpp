/**
 * @file
 */
#pragma once

#include "src/type/Type.hpp"
#include "src/common/Named.hpp"
#include "src/common/TypeArgumented.hpp"

namespace birch {
/**
 * Name in the context of an type, referring to a basic, class or generic
 * type.
 *
 * @ingroup type
 */
class NamedType: public Type, public Named, public TypeArgumented {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param typeArgs Generic type arguments.
   * @param loc Location.
   */
  NamedType(Name* name, Type* typeArgs, Location* loc = nullptr);

  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   */
  NamedType(Name* name, Location* loc = nullptr);

  virtual void accept(Visitor* visitor) const;
};
}
