/**
 * @file
 */
#include "bi/expression/Expression.hpp"

#include "bi/expression/ExpressionIterator.hpp"
#include "bi/expression/ExpressionConstIterator.hpp"
#include "bi/type/EmptyType.hpp"
#include "bi/expression/Range.hpp"
#include "bi/visitor/all.hpp"

bi::Expression::Expression(Location* loc) :
    Located(loc) {
  //
}

bi::Expression::~Expression() {
  //
}

bool bi::Expression::isEmpty() const {
  return false;
}

bool bi::Expression::isAssignable() const {
  return false;
}

bool bi::Expression::isSlice() const {
  return false;
}

bool bi::Expression::isTuple() const {
  return false;
}

bool bi::Expression::isMember() const {
  return false;
}

const bi::Expression* bi::Expression::strip() const {
  return this;
}

int bi::Expression::width() const {
  int result = 0;
  for (auto iter = begin(); iter != end(); ++iter) {
    ++result;
  }
  return result;
}

int bi::Expression::depth() const {
  int result = 0;
  for (auto iter = begin(); iter != end(); ++iter) {
    if (dynamic_cast<const Range*>(*iter)) {
      ++result;
    }
  }
  return result;
}

bi::ExpressionIterator bi::Expression::begin() {
  if (isEmpty()) {
    return end();
  } else {
    return ExpressionIterator(this);
  }
}

bi::ExpressionIterator bi::Expression::end() {
  return ExpressionIterator(nullptr);
}

bi::ExpressionConstIterator bi::Expression::begin() const {
  if (isEmpty()) {
    return end();
  } else {
    return ExpressionConstIterator(this);
  }
}

bi::ExpressionConstIterator bi::Expression::end() const {
  return ExpressionConstIterator(nullptr);
}
