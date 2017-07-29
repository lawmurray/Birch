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
struct Unknown {
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
  Identifier(shared_ptr<Name> name, shared_ptr<Location> loc = nullptr,
      ObjectType* target = nullptr);

  /**
   * Destructor.
   */
  virtual ~Identifier();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const Identifier<ObjectType>& o) const;
  virtual bool definitely(const Parameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const Identifier<ObjectType>& o) const;
  virtual bool possibly(const Parameter& o) const;
};
}
