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

  virtual bool isRandom() const;

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(Type& o);
  virtual bool definitely(EmptyType& o);
  virtual bool definitely(LambdaType& o);
  virtual bool definitely(List<Type>& o);
  virtual bool definitely(ModelParameter& o);
  virtual bool definitely(ModelReference& o);

  virtual bool dispatchPossibly(Type& o);
  virtual bool possibly(EmptyType& o);
  virtual bool possibly(LambdaType& o);
  virtual bool possibly(List<Type>& o);
  virtual bool possibly(ModelParameter& o);
  virtual bool possibly(ModelReference& o);
  virtual bool possibly(RandomType& o);

  virtual bool equals(Type& o);
};
}
