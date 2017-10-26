/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Reference.hpp"
#include "bi/statement/Generic.hpp"

namespace bi {
/**
 * Generic type.
 *
 * @ingroup compiler_type
 */
class GenericType: public Type,
    public Named,
    public Reference<Generic> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   * @param assignable Is this type assignable?
   * @param target Target.
   */
  GenericType(Name* name, Location* loc = nullptr,
      const bool assignable = false, Generic* target = nullptr);

  /**
   * Constructor.
   *
   * @param target Target.
   */
  GenericType(Generic* target);

  /**
   * Destructor.
   */
  virtual ~GenericType();

  virtual bool isGeneric() const;

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const AliasType& o) const;
  virtual bool definitely(const GenericType& o) const;
  virtual bool definitely(const OptionalType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const AliasType& o) const;
  virtual bool possibly(const GenericType& o) const;
  virtual bool possibly(const OptionalType& o) const;
};
}
