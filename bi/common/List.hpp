/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/statement/Statement.hpp"
#include "bi/type/Type.hpp"
#include "bi/primitive/unique_ptr.hpp"

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
  explicit List(T* head, T* tail, shared_ptr<Location> loc = nullptr);

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

  virtual T* acceptClone(Cloner* visitor) const;
  virtual void acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool operator<=(T& o);
  virtual bool operator==(const T& o) const;

  /**
   * Left operand.
   */
  unique_ptr<T> head;

  /**
   * Right operand.
   */
  unique_ptr<T> tail;
};

/**
 * Expression list.
 *
 * @ingroup compiler_common
 */
typedef List<Expression> ExpressionList;

/**
 * Statement list.
 *
 * @ingroup compiler_common
 */
typedef List<Statement> StatementList;

/**
 * Type list.
 *
 * @ingroup compiler_common
 */
typedef List<Type> TypeList;
}

template<class T>
inline bi::List<T>::~List() {
  //
}
