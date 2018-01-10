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
 * @ingroup birch_type
 */
class PointerType: public Type, public Single<Type> {
public:
  /**
   * Constructor.
   *
   * @param weak Is this a weak pointer type?
   * @param single Type.
   * @param read Is this a read-only pointer type?
   * @param loc Location.
   */
  PointerType(const bool weak, Type* single, const bool read, Location* loc =
      nullptr);

  /**
   * Destructor.
   */
  virtual ~PointerType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isPointer() const;

  virtual Type* unwrap();
  virtual const Type* unwrap() const;

  using Type::definitely;
  using Type::common;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const GenericType& o) const;
  virtual bool definitely(const OptionalType& o) const;
  virtual bool definitely(const PointerType& o) const;
  virtual bool definitely(const AnyType& o) const;

  virtual Type* dispatchCommon(const Type& o) const;
  virtual Type* common(const GenericType& o) const;
  virtual Type* common(const OptionalType& o) const;
  virtual Type* common(const PointerType& o) const;
  virtual Type* common(const AnyType& o) const;

  /**
   * Is this a weak pointer type?
   */
  bool weak;

  /**
   * Is this a read-only pointer type?
   */
  bool read;

};
}
