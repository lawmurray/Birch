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
  RandomType(Type* left, Type* right, shared_ptr<Location> loc =
      nullptr);

  /**
   * Destructor.
   */
  virtual ~RandomType();

  virtual Type* acceptClone(Cloner* visitor) const;
  virtual Type* acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool operator<=(Type& o);
  virtual bool operator==(const Type& o) const;
};
}
