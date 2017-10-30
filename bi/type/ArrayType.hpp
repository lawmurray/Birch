/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Bracketed.hpp"
#include "bi/common/Single.hpp"

namespace bi {
/**
 * Array type.
 *
 * @ingroup compiler_type
 */
class ArrayType: public Type, public Single<Type>, public Bracketed {
public:
  /**
   * Constructor.
   *
   * @param single Type.
   * @param brackets Brackets.
   * @param loc Location.
   * @param assignable Is this type assignable?
   */
  ArrayType(Type* single, Expression* brackets, Location* loc =
      nullptr, const bool assignable = false);

  /**
   * Constructor.
   *
   * @param single Type.
   * @param ndims Number of dimensions.
   * @param loc Location.
   * @param assignable Is this type assignable?
   */
  ArrayType(Type* single, const int ndims, Location* loc =
      nullptr, const bool assignable = false);

  /**
   * Destructor.
   */
  virtual ~ArrayType();

  virtual int dims() const;
  virtual bool isArray() const;

  virtual void resolveConstructor(Argumented* o);

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const ArrayType& o) const;
  virtual bool definitely(const AliasType& o) const;
  virtual bool definitely(const GenericType& o) const;
  virtual bool definitely(const OptionalType& o) const;
  virtual bool definitely(const EmptyType& o) const;

  /**
   * Number of dimensions.
   */
  int ndims;
};
}
