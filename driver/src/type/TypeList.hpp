/**
 * @file
 */
#pragma once

#include "src/type/Type.hpp"

namespace birch {
/**
 * List type.
 *
 * @ingroup type
 */
class TypeList: public Type {
public:
  /**
   * Constructor.
   *
   * @param head First in list.
   * @param tail Remaining list.
   * @param loc Location.
   */
  TypeList(Type* head, Type* tail, Location* loc = nullptr);

  virtual void accept(Visitor* visitor) const;

  /**
   * Left operand.
   */
  Type* head;

  /**
   * Right operand.
   */
  Type* tail;
};
}
