/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/statement/Statement.hpp"
#include "bi/type/Type.hpp"
#include "bi/common/Iterator.hpp"

namespace bi {
/**
 * List.
 *
 * @ingroup compiler_common
 */
template<class T>
class List: public T {
public:
  /**
   * Constructor.
   *
   * @param head First in list.
   * @param tail Remaining list.
   * @param loc Location.
   */
  List(T* head, T* tail, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~List();

  /**
   * Number of objects in the list.
   */
  virtual int count() const;

  /**
   * Number of Range objects in the list.
   */
  virtual int rangeCount() const;

  virtual T* accept(Cloner* visitor) const;
  virtual T* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Left operand.
   */
  T* head;

  /**
   * Right operand.
   */
  T* tail;
};
}
