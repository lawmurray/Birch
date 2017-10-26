/**
 * @file
 */
#include "bi/expression/ExpressionIterator.hpp"

#include "bi/expression/ExpressionList.hpp"

bi::ExpressionIterator::ExpressionIterator(const Expression* o) :
    o(o) {
  //
}

bi::ExpressionIterator& bi::ExpressionIterator::operator++() {
  auto list = dynamic_cast<const ExpressionList*>(o);
  if (list) {
    o = list->tail;
  } else {
    o = nullptr;
  }
  return *this;
}

bi::ExpressionIterator bi::ExpressionIterator::operator++(int) {
  ExpressionIterator result = *this;
  ++*this;
  return result;
}

const bi::Expression* bi::ExpressionIterator::operator*() {
  auto list = dynamic_cast<const ExpressionList*>(o);
  if (list) {
    return list->head;
  } else {
    return o;
  }
}

bool bi::ExpressionIterator::operator==(const ExpressionIterator& o) const {
  return this->o == o.o;
}

bool bi::ExpressionIterator::operator!=(const ExpressionIterator& o) const {
  return this->o != o.o;
}
