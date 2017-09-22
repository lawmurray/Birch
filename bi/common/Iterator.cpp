/**
 * @file
 */
#include "bi/common/Iterator.hpp"

#include "bi/common/List.hpp"
#include "bi/type/ListType.hpp"

template<class T>
bi::Iterator<T>::Iterator(const T* o) : o(o) {
  //
}

template<class T>
bi::Iterator<T>& bi::Iterator<T>::operator++() {
  const List<T>* list = dynamic_cast<const List<T>*>(o);
  if (list) {
    o = list->tail;
  } else {
    const ListType* list = dynamic_cast<const ListType*>(o);
    if (list) {
      o = reinterpret_cast<T*>(list->tail);
    } else {
      o = nullptr;
    }
  }
  return *this;
}

template<class T>
bi::Iterator<T> bi::Iterator<T>::operator++(int) {
  bi::Iterator<T> result = *this;
  ++*this;
  return result;
}

template<class T>
const T* bi::Iterator<T>::operator*() {
  const List<T>* list = dynamic_cast<const List<T>*>(o);
  if (list) {
    return list->head;
  } else {
    const ListType* list = dynamic_cast<const ListType*>(o);
    if (list) {
      return reinterpret_cast<T*>(list->head);
    } else {
      return o;
    }
  }
}

template<class T>
bool bi::Iterator<T>::operator==(const Iterator<T>& o) const {
  return this->o == o.o;
}

template<class T>
bool bi::Iterator<T>::operator!=(const Iterator<T>& o) const {
  return this->o != o.o;
}

/*
 * Explicit template instantiations.
 */
template class bi::Iterator<bi::Expression>;
template class bi::Iterator<bi::Statement>;
template class bi::Iterator<bi::Type>;
