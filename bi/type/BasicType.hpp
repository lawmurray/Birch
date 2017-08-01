/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Reference.hpp"
#include "bi/statement/Basic.hpp"

namespace bi {
/**
 * Basic type.
 *
 * @ingroup compiler_type
 *
 * @tparam ObjectType The particular type of object referred to by the
 * identifier.
 */
class BasicType: public Type,
    public Named,
    public Reference<Basic> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   * @param assignable Is this type assignable?
   * @param target Target.
   */
  BasicType(Name* name, Location* loc = nullptr,
      const bool assignable = false, Basic* target = nullptr);

  /**
   * Constructor.
   *
   * @param target Target.
   */
  BasicType(Basic* target);

  /**
   * Destructor.
   */
  virtual ~BasicType();

  virtual bool isBasic() const;

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const AliasType& o) const;
  virtual bool definitely(const BasicType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const AliasType& o) const;
  virtual bool possibly(const BasicType& o) const;
};
}
