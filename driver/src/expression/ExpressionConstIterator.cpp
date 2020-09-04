/**
 * @file
 */
#include "src/expression/ExpressionConstIterator.hpp"

#include "src/expression/ExpressionList.hpp"

birch::ExpressionConstIterator::ExpressionConstIterator(const Expression* o) :
    o(o) {
  //
}

birch::ExpressionConstIterator& birch::ExpressionConstIterator::operator++() {
  auto list = dynamic_cast<const ExpressionList*>(o);
  if (list) {
    o = list->tail;
  } else {
    o = nullptr;
  }
  return *this;
}

birch::ExpressionConstIterator birch::ExpressionConstIterator::operator++(int) {
  ExpressionConstIterator result = *this;
  ++*this;
  return result;
}

const birch::Expression* birch::ExpressionConstIterator::operator*() {
  auto list = dynamic_cast<const ExpressionList*>(o);
  if (list) {
    return list->head;
  } else {
    return o;
  }
}

bool birch::ExpressionConstIterator::operator==(const ExpressionConstIterator& o) const {
  return this->o == o.o;
}

bool birch::ExpressionConstIterator::operator!=(const ExpressionConstIterator& o) const {
  return this->o != o.o;
}
