/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Single.hpp"

namespace bi {
/**
 * Sequence type.
 *
 * @ingroup type
 */
class SequenceType: public Type, public Single<Type> {
public:
  /**
   * Constructor.
   *
   * @param single Type of all elements of the sequence.
   * @param loc Location.
   */
  SequenceType(Type* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~SequenceType();

  virtual int depth() const;
  virtual Type* element();
  virtual const Type* element() const;
  virtual bool isSequence() const;

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;
  using Type::common;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const ArrayType& o) const;
  virtual bool definitely(const GenericType& o) const;
  virtual bool definitely(const OptionalType& o) const;
  virtual bool definitely(const SequenceType& o) const;

  virtual Type* dispatchCommon(const Type& o) const;
  virtual Type* common(const ArrayType& o) const;
  virtual Type* common(const GenericType& o) const;
  virtual Type* common(const OptionalType& o) const;
  virtual Type* common(const SequenceType& o) const;
};
}
