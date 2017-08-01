/**
 * @file
 */
#include "bi/common/List.hpp"

#include "bi/visitor/all.hpp"

template<class T>
bi::List<T>::List(T* head, T* tail, shared_ptr<Location> loc) :
    head(head),
    tail(tail) {
  /* pre-conditions */
  assert(head);
  assert(tail);

  this->loc = loc;
}

template<class T>
bi::List<T>::~List() {
  //
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
T* bi::List<T>::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

template<class T>
T* bi::List<T>::accept(Modifier* visitor) {
  return visitor->modify(this);
}

template<class T>
void bi::List<T>::accept(Visitor* visitor) const {
  visitor->visit(this);
}

/*
 * Explicit template instantiations.
 */
template class bi::List<bi::Expression>;
template class bi::List<bi::Statement>;
