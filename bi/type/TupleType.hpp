/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Single.hpp"

namespace bi {
/**
 * Parentheses type.
 *
 * @ingroup compiler_type
 */
class TupleType: public Type, public Single<Type> {
public:
  /**
   * Constructor.
   *
   * @param single Type in parentheses.
   * @param loc Location.
   * @param assignable Is this type assignable?
   */
  TupleType(Type* single, Location* loc = nullptr,
      const bool assignable = false);

  /**
   * Destructor.
   */
  virtual ~TupleType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const AliasType& o) const;
  virtual bool definitely(const OptionalType& o) const;
  virtual bool definitely(const TupleType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const AliasType& o) const;
  virtual bool possibly(const OptionalType& o) const;
  virtual bool possibly(const TupleType& o) const;
};
}
