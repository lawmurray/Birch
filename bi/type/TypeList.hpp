/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"

namespace bi {
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

  /**
   * Destructor.
   */
  virtual ~TypeList();

  virtual bool isList() const;

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Left operand.
   */
  Type* head;

  /**
   * Right operand.
   */
  Type* tail;

  using Type::isConvertible;
  using Type::isAssignable;

  virtual bool dispatchIsConvertible(const Type& o) const;
  virtual bool isConvertible(const TypeList& o) const;

  virtual bool dispatchIsAssignable(const Type& o) const;
  virtual bool isAssignable(const TypeList& o) const;
};
}
