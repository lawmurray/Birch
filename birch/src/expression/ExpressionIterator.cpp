/**
 * @file
 */
#include "src/expression/ExpressionIterator.hpp"

#include "src/expression/ExpressionList.hpp"

birch::ExpressionIterator::ExpressionIterator(Expression* o) :
    o(o) {
  //
}

birch::ExpressionIterator& birch::ExpressionIterator::operator++() {
  auto list = dynamic_cast<ExpressionList*>(o);
  if (list) {
    o = list->tail;
  } else {
    o = nullptr;
  }
  return *this;
}

birch::ExpressionIterator birch::ExpressionIterator::operator++(int) {
  ExpressionIterator result = *this;
  ++*this;
  return result;
}

birch::Expression* birch::ExpressionIterator::operator*() {
  auto list = dynamic_cast<ExpressionList*>(o);
  if (list) {
    return list->head;
  } else {
    return o;
  }
}

bool birch::ExpressionIterator::operator==(const ExpressionIterator& o) const {
  return this->o == o.o;
}

bool birch::ExpressionIterator::operator!=(const ExpressionIterator& o) const {
  return this->o != o.o;
}
