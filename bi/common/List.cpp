/**
 * @file
 */
#include "bi/common/List.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>
#include <type_traits>

template<class T>
bi::List<T>::List(T* head, T* tail,
    shared_ptr<Location> loc) :
    head(head), tail(tail) {
  /* pre-conditions */
  assert(head);
  assert(tail);

  this->loc = loc;
}

template<class T>
int bi::List<T>::count() const {
  const List<T>* listTail = dynamic_cast<const List<T>*>(tail.get());
  if (listTail) {
    return 1 + listTail->count();
  } else {
    return 2;
  }
}

template<class T>
int bi::List<T>::rangeCount() const {
  const Range* rangeHead = dynamic_cast<const Range*>(head.get());
  const Range* rangeTail = dynamic_cast<const Range*>(tail.get());
  const List<T>* listTail = dynamic_cast<const List<T>*>(tail.get());
  int count = 0;

  if (rangeHead) {
    ++count;
  }
  if (rangeTail) {
    ++count;
  } else if (listTail) {
    count += listTail->rangeCount();
  }
  return count;
}

template<class T>
T* bi::List<T>::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

template<class T>
void bi::List<T>::acceptModify(Modifier* visitor) {
  visitor->modify(this);
}

template<class T>
void bi::List<T>::accept(Visitor* visitor) const {
  visitor->visit(this);
}

template<class T>
bool bi::List<T>::operator<=(T& o) {
  try {
    List<T>& o1 = dynamic_cast<List<T>&>(o);
    return *head <= *o1.head && *tail <= *o1.tail;
  } catch (std::bad_cast e) {
    //
  }
  if (std::is_same<T,Expression>::value) {
    try {
      ParenthesesExpression& o1 = dynamic_cast<ParenthesesExpression&>(o);
      return *this <= *o1.expr;
    } catch (std::bad_cast e) {
      //
    }
  }
  return false;
}

template<class T>
bool bi::List<T>::operator==(const T& o) const {
  try {
    const List<T>& o1 = dynamic_cast<const List<T>&>(o);
    return *head == *o1.head && *tail == *o1.tail;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

/*
 * Explicit template instantiations.
 */
template class bi::List<bi::Expression>;
template class bi::List<bi::Statement>;
template class bi::List<bi::Type>;
