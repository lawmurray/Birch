/**
 * @file
 */
#pragma once

#include "bi/expression/Identifier.hpp"
#include "bi/common/Overloaded.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/TypeArgumented.hpp"
#include "bi/common/Reference.hpp"

namespace bi {
/**
 * Identifier for overloaded function.
 *
 * @ingroup expression
 *
 * @tparam ObjectType The particular type of object referred to by the
 * identifier.
 */
template<class ObjectType>
class OverloadedIdentifier: public Expression,
    public Named,
    public TypeArgumented,
    public Reference<Overloaded<ObjectType>> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param typeArgs Generic type arguments.
   * @param loc Location.
   * @param target Target.
   */
  OverloadedIdentifier(Name* name, Type* typeArgs, Location* loc = nullptr,
      Overloaded<ObjectType>* target = nullptr);

  /**
   * Destructor.
   */
  virtual ~OverloadedIdentifier();

  virtual bool isOverloaded() const;

  virtual FunctionType* resolve(Argumented* o);

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Inherited objects that may be an alternative to target.
   */
  std::list<Overloaded<ObjectType>*> inherited;
};
}
