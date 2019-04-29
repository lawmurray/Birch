/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Single.hpp"

namespace bi {
/**
 * Pointer type.
 *
 * @ingroup type
 */
class WeakType: public Type, public Single<Type> {
public:
  /**
   * Constructor.
   *
   * @param single Type.
   * @param loc Location.
   */
  WeakType(Type* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~WeakType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isWeak() const;

  virtual Type* unwrap();
  virtual const Type* unwrap() const;

  virtual void resolveConstructor(Argumented* args);

  using Type::isConvertible;
  using Type::common;

  virtual bool dispatchIsConvertible(const Type& o) const;
  virtual bool isConvertible(const GenericType& o) const;
  virtual bool isConvertible(const MemberType& o) const;
  virtual bool isConvertible(const OptionalType& o) const;
  virtual bool isConvertible(const WeakType& o) const;

  virtual Type* dispatchCommon(const Type& o) const;
  virtual Type* common(const ClassType& o) const;
  virtual Type* common(const GenericType& o) const;
  virtual Type* common(const MemberType& o) const;
  virtual Type* common(const OptionalType& o) const;
  virtual Type* common(const WeakType& o) const;
};
}
