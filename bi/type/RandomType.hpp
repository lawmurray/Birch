/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Binary.hpp"

namespace bi {
/**
 * Random variable type.
 *
 * @ingroup compiler_type
 */
class RandomType: public Type, public TypeBinary {
public:
  /**
   * Constructor.
   *
   * @param left Variate type.
   * @param right Model type.
   * @param loc Location.
   */
  RandomType(Type* left, Type* right, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~RandomType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual possibly dispatch(Type& o);
  virtual possibly le(AssignableType& o);
  virtual possibly le(ParenthesesType& o);
  virtual possibly le(EmptyType& o);
  virtual possibly le(List<Type>& o);
  virtual possibly le(ModelParameter& o);
  virtual possibly le(ModelReference& o);
  virtual possibly le(RandomType& o);
};
}
