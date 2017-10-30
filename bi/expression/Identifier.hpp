/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Reference.hpp"

namespace bi {
/**
 * Placeholder for identifier to unknown object.
 */
class Unknown {
  //
};

/**
 * Identifier.
 *
 * @ingroup compiler_expression
 *
 * @tparam ObjectType The particular type of object referred to by the
 * identifier.
 *
 * An identifier refers to a variable, function or the like. For most
 * identifiers, the type of object referred to is ambiguous in syntax:
 * for example, @c f() may call a function or a variable of lambda type. When
 * parsing, such identifiers are given the type @c Identifier<Unknown>, and
 * are later replaced, by Resolver, with an identifier of the appropriate
 * type.
 */
template<class ObjectType = Unknown>
class Identifier: public Expression,
    public Named,
    public Reference<ObjectType> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   * @param target Target.
   */
  Identifier(Name* name, Location* loc = nullptr,
      ObjectType* target = nullptr);

  /**
   * Destructor.
   */
  virtual ~Identifier();

  virtual bool isAssignable() const;

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
