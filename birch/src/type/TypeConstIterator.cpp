/**
 * @file
 */
#include "src/type/TypeConstIterator.hpp"

#include "src/type/TypeList.hpp"

birch::TypeConstIterator::TypeConstIterator(const Type* o) :
    o(o) {
  //
}

birch::TypeConstIterator& birch::TypeConstIterator::operator++() {
  auto list = dynamic_cast<const TypeList*>(o);
  if (list) {
    o = list->tail;
  } else {
    o = nullptr;
  }
  return *this;
}

birch::TypeConstIterator birch::TypeConstIterator::operator++(int) {
  TypeConstIterator result = *this;
  ++*this;
  return result;
}

const birch::Type* birch::TypeConstIterator::operator*() {
  auto list = dynamic_cast<const TypeList*>(o);
  if (list) {
    return list->head;
  } else {
    return o;
  }
}

bool birch::TypeConstIterator::operator==(const TypeConstIterator& o) const {
  return this->o == o.o;
}

bool birch::TypeConstIterator::operator!=(const TypeConstIterator& o) const {
  return this->o != o.o;
}
