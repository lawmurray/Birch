/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/TypeArgumented.hpp"
#include "bi/common/Scope.hpp"

namespace bi {
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

  /**
   * Constructor.
   *
   * @param target Class to reference.
   * @param loc Location.
   *
   * Name and generic type arguments are determined from the target argument.
   */
  NamedType(Class* target, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~NamedType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isBasic() const;
  virtual bool isClass() const;
  virtual bool isGeneric() const;
  virtual bool isValue() const;

  /**
   * The category of the identifier.
   */
  TypeCategory category;

  /**
   * Once resolved, the unique number of the referent.
   */
  int number;
};
}
