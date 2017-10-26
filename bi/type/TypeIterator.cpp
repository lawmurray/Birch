/**
 * @file
 */
#include "bi/type/TypeIterator.hpp"

#include "bi/type/TypeList.hpp"

bi::TypeIterator::TypeIterator(const Type* o) :
    o(o) {
  //
}

bi::TypeIterator& bi::TypeIterator::operator++() {
  auto list = dynamic_cast<const TypeList*>(o);
  if (list) {
    o = list->tail;
  } else {
    o = nullptr;
  }
  return *this;
}

bi::TypeIterator bi::TypeIterator::operator++(int) {
  TypeIterator result = *this;
  ++*this;
  return result;
}

const bi::Type* bi::TypeIterator::operator*() {
  auto list = dynamic_cast<const TypeList*>(o);
  if (list) {
    return list->head;
  } else {
    return o;
  }
}

bool bi::TypeIterator::operator==(const TypeIterator& o) const {
  return this->o == o.o;
}

bool bi::TypeIterator::operator!=(const TypeIterator& o) const {
  return this->o != o.o;
}
