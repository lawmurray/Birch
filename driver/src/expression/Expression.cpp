/**
 * @file
 */
#include "src/expression/Expression.hpp"

#include "src/expression/ExpressionIterator.hpp"
#include "src/expression/ExpressionConstIterator.hpp"
#include "src/type/EmptyType.hpp"
#include "src/expression/Range.hpp"
#include "src/visitor/all.hpp"

birch::Expression::Expression(Location* loc) :
    Located(loc) {
  //
}

bool birch::Expression::isEmpty() const {
  return false;
}

bool birch::Expression::isSlice() const {
  return false;
}

bool birch::Expression::isTuple() const {
  return false;
}

bool birch::Expression::isMembership() const {
  return false;
}

bool birch::Expression::isThis() const {
  return false;
}

bool birch::Expression::isSuper() const {
  return false;
}

bool birch::Expression::isGlobal() const {
  return false;
}

bool birch::Expression::isMember() const {
  return false;
}

bool birch::Expression::isLocal() const {
  return false;
}

bool birch::Expression::isParameter() const {
  return false;
}

const birch::Expression* birch::Expression::strip() const {
  return this;
}

int birch::Expression::width() const {
  int result = 0;
  for (auto iter = begin(); iter != end(); ++iter) {
    ++result;
  }
  return result;
}

int birch::Expression::depth() const {
  int result = 0;
  for (auto iter = begin(); iter != end(); ++iter) {
    if (dynamic_cast<const Range*>(*iter)) {
      ++result;
    }
  }
  return result;
}

birch::ExpressionIterator birch::Expression::begin() {
  if (isEmpty()) {
    return end();
  } else {
    return ExpressionIterator(this);
  }
}

birch::ExpressionIterator birch::Expression::end() {
  return ExpressionIterator(nullptr);
}

birch::ExpressionConstIterator birch::Expression::begin() const {
  if (isEmpty()) {
    return end();
  } else {
    return ExpressionConstIterator(this);
  }
}

birch::ExpressionConstIterator birch::Expression::end() const {
  return ExpressionConstIterator(nullptr);
}
