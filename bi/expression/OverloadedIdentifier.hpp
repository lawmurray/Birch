/**
 * @file
 */
#pragma once

#include "bi/expression/Identifier.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Overloaded.hpp"
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
  OverloadedIdentifier(shared_ptr<Name> name, shared_ptr<Location> loc =
      nullptr, const Overloaded<ObjectType>* target = nullptr);

  /**
   * Destructor.
   */
  virtual ~OverloadedIdentifier();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const OverloadedIdentifier<ObjectType>& o) const;
  virtual bool definitely(const Parameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const OverloadedIdentifier<ObjectType>& o) const;
  virtual bool possibly(const Parameter& o) const;
};
}
