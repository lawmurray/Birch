/**
 * @file
 */
#include "bi/statement/StatementIterator.hpp"

#include "bi/statement/StatementList.hpp"

bi::StatementIterator::StatementIterator(const Statement* o) :
    o(o) {
  //
}

bi::StatementIterator& bi::StatementIterator::operator++() {
  auto list = dynamic_cast<const StatementList*>(o);
  if (list) {
    o = list->tail;
  } else {
    o = nullptr;
  }
  return *this;
}

bi::StatementIterator bi::StatementIterator::operator++(int) {
  StatementIterator result = *this;
  ++*this;
  return result;
}

const bi::Statement* bi::StatementIterator::operator*() {
  auto list = dynamic_cast<const StatementList*>(o);
  if (list) {
    return list->head;
  } else {
    return o;
  }
}

bool bi::StatementIterator::operator==(const StatementIterator& o) const {
  return this->o == o.o;
}

bool bi::StatementIterator::operator!=(const StatementIterator& o) const {
  return this->o != o.o;
}
