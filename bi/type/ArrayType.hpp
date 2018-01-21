/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Single.hpp"

namespace bi {
/**
 * Array type.
 *
 * @ingroup type
 */
class ArrayType: public Type, public Single<Type> {
public:
  /**
   * Constructor.
   *
   * @param single Type.
   * @param ndims Number of dimensions.
   * @param loc Location.
   */
  ArrayType(Type* single, const int ndims, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~ArrayType();

  virtual int depth() const;
  virtual Type* element();
  virtual const Type* element() const;
  virtual bool isArray() const;
  virtual void resolveConstructor(Argumented* o);

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;
  using Type::common;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const ArrayType& o) const;
  virtual bool definitely(const GenericType& o) const;
  virtual bool definitely(const OptionalType& o) const;
  virtual bool definitely(const AnyType& o) const;

  virtual Type* dispatchCommon(const Type& o) const;
  virtual Type* common(const ArrayType& o) const;
  virtual Type* common(const GenericType& o) const;
  virtual Type* common(const OptionalType& o) const;
  virtual Type* common(const AnyType& o) const;

  /**
   * Number of dimensions.
   */
  int ndims;
};
}
