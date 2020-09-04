/**
 * @file
 */
#pragma once

#include "src/type/Type.hpp"
#include "src/common/Single.hpp"

namespace birch {
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

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual int depth() const;
  virtual Type* element();
  virtual const Type* element() const;
  virtual bool isArray() const;
  virtual bool isValue() const;

  /**
   * Number of dimensions.
   */
  int ndims;
};
}
