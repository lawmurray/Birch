/**
 * @file
 */
#pragma once

#include "bi/expression/Identifier.hpp"
#include "bi/common/Overloaded.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Reference.hpp"

namespace bi {
/**
 * Identifier for overloaded function.
 *
 * @ingroup compiler_expression
 *
 * @tparam ObjectType The particular type of object referred to by the
 * identifier.
 */
template<class ObjectType>
class OverloadedIdentifier: public Expression, public Named, public Reference<
    Overloaded<ObjectType>> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   * @param target Target.
   */
  OverloadedIdentifier(Name* name, Location* loc = nullptr,
      Overloaded<ObjectType>* target = nullptr);

  /**
   * Destructor.
   */
  virtual ~OverloadedIdentifier();

  virtual bool isOverloaded() const;

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
