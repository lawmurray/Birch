/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Binary.hpp"
#include "bi/common/Scoped.hpp"

namespace bi {
/**
 * Random variable type.
 *
 * @ingroup compiler_type
 */
class RandomType: public Type, public Scoped, public TypeBinary {
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

  /**
   * Member variable for the variate.
   */
  unique_ptr<Expression> x;

  /**
   * Member variable for the model.
   */
  unique_ptr<Expression> m;

  virtual bool dispatch(Type& o);
  virtual bool le(EmptyType& o);
  virtual bool le(List<Type>& o);
  virtual bool le(ModelParameter& o);
  virtual bool le(ModelReference& o);
  virtual bool le(RandomType& o);
};
}
