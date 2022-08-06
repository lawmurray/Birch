/**
 * @file
 */
#include "src/statement/StatementIterator.hpp"

#include "src/statement/StatementList.hpp"

birch::StatementIterator::StatementIterator(const Statement* o) :
    o(o) {
  //
}

birch::StatementIterator& birch::StatementIterator::operator++() {
  auto list = dynamic_cast<const StatementList*>(o);
  if (list) {
    o = list->tail;
  } else {
    o = nullptr;
  }
  return *this;
}

birch::StatementIterator birch::StatementIterator::operator++(int) {
  StatementIterator result = *this;
  ++*this;
  return result;
}

const birch::Statement* birch::StatementIterator::operator*() {
  auto list = dynamic_cast<const StatementList*>(o);
  if (list) {
    return list->head;
  } else {
    return o;
  }
}

bool birch::StatementIterator::operator==(const StatementIterator& o) const {
  return this->o == o.o;
}

bool birch::StatementIterator::operator!=(const StatementIterator& o) const {
  return this->o != o.o;
}
