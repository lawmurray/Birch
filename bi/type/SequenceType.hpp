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
 * @ingroup compiler_type
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

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const AliasType& o) const;
  virtual bool definitely(const ArrayType& o) const;
  virtual bool definitely(const GenericType& o) const;
  virtual bool definitely(const OptionalType& o) const;
  virtual bool definitely(const SequenceType& o) const;
  virtual bool definitely(const AnyType& o) const;
};
}
