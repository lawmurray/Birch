/**
 * @file
 */
#include "src/type/TypeIterator.hpp"

#include "src/type/TypeList.hpp"

birch::TypeIterator::TypeIterator(Type* o) :
    o(o) {
  //
}

birch::TypeIterator& birch::TypeIterator::operator++() {
  auto list = dynamic_cast<const TypeList*>(o);
  if (list) {
    o = list->tail;
  } else {
    o = nullptr;
  }
  return *this;
}

birch::TypeIterator birch::TypeIterator::operator++(int) {
  TypeIterator result = *this;
  ++*this;
  return result;
}

birch::Type* birch::TypeIterator::operator*() {
  auto list = dynamic_cast<const TypeList*>(o);
  if (list) {
    return list->head;
  } else {
    return o;
  }
}

bool birch::TypeIterator::operator==(const TypeIterator& o) const {
  return this->o == o.o;
}

bool birch::TypeIterator::operator!=(const TypeIterator& o) const {
  return this->o != o.o;
}
