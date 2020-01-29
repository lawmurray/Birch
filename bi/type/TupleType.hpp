/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Single.hpp"

namespace bi {
/**
 * Tuple type.
 *
 * @ingroup type
 */
class TupleType: public Type, public Single<Type> {
public:
  /**
   * Constructor.
   *
   * @param single Type in parentheses.
   * @param loc Location.
   */
  TupleType(Type* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~TupleType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::isConvertible;
  using Type::isAssignable;

  virtual bool dispatchIsConvertible(const Type& o) const;
  virtual bool isConvertible(const GenericType& o) const;
  virtual bool isConvertible(const MemberType& o) const;
  virtual bool isConvertible(const OptionalType& o) const;
  virtual bool isConvertible(const TupleType& o) const;

  virtual bool dispatchIsAssignable(const Type& o) const;
  virtual bool isAssignable(const ClassType& o) const;
  virtual bool isAssignable(const GenericType& o) const;
  virtual bool isAssignable(const MemberType& o) const;
  virtual bool isAssignable(const OptionalType& o) const;
  virtual bool isAssignable(const TupleType& o) const;
};
}
