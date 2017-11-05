/**
 * @file
 */
#include "bi/type/TypeConstIterator.hpp"

#include "bi/type/TypeList.hpp"

bi::TypeConstIterator::TypeConstIterator(const Type* o) :
    o(o) {
  //
}

bi::TypeConstIterator& bi::TypeConstIterator::operator++() {
  auto list = dynamic_cast<const TypeList*>(o);
  if (list) {
    o = list->tail;
  } else {
    o = nullptr;
  }
  return *this;
}

bi::TypeConstIterator bi::TypeConstIterator::operator++(int) {
  TypeConstIterator result = *this;
  ++*this;
  return result;
}

const bi::Type* bi::TypeConstIterator::operator*() {
  auto list = dynamic_cast<const TypeList*>(o);
  if (list) {
    return list->head;
  } else {
    return o;
  }
}

bool bi::TypeConstIterator::operator==(const TypeConstIterator& o) const {
  return this->o == o.o;
}

bool bi::TypeConstIterator::operator!=(const TypeConstIterator& o) const {
  return this->o != o.o;
}
