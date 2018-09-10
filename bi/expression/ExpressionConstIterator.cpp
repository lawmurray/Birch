/**
 * @file
 */
#include "bi/expression/ExpressionConstIterator.hpp"

#include "bi/expression/ExpressionList.hpp"

bi::ExpressionConstIterator::ExpressionConstIterator(const Expression* o) :
    o(o) {
  //
}

bi::ExpressionConstIterator& bi::ExpressionConstIterator::operator++() {
  auto list = dynamic_cast<const ExpressionList*>(o);
  if (list) {
    o = list->tail;
  } else {
    o = nullptr;
  }
  return *this;
}

bi::ExpressionConstIterator bi::ExpressionConstIterator::operator++(int) {
  ExpressionConstIterator result = *this;
  ++*this;
  return result;
}

const bi::Expression* bi::ExpressionConstIterator::operator*() {
  auto list = dynamic_cast<const ExpressionList*>(o);
  if (list) {
    return list->head;
  } else {
    return o;
  }
}

bool bi::ExpressionConstIterator::operator==(const ExpressionConstIterator& o) const {
  return this->o == o.o;
}

bool bi::ExpressionConstIterator::operator!=(const ExpressionConstIterator& o) const {
  return this->o != o.o;
}
