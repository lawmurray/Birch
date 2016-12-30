/**
 * @file
 */
#include "bi/expression/Expression.hpp"

#include "bi/type/EmptyType.hpp"
#include "bi/visitor/IsPrimary.hpp"
#include "bi/visitor/IsRich.hpp"
#include "bi/visitor/TupleSizer.hpp"

bi::Expression::Expression(Type* type, shared_ptr<Location> loc) :
    Located(loc), Typed(type) {
  //
}

bi::Expression::Expression(shared_ptr<Location> loc) :
    Located(loc) {
  //
}

bi::Expression::~Expression() {
  //
}

bool bi::Expression::isPrimary() const {
  IsPrimary visitor;
  this->accept(&visitor);
  return visitor.result;
}

bool bi::Expression::isRich() const {
  IsRich visitor;
  this->accept(&visitor);
  return visitor.result;
}

bi::Expression* bi::Expression::strip() {
  return this;
}

int bi::Expression::tupleSize() const {
  TupleSizer visitor;
  this->accept(&visitor);
  return visitor.size;
}

int bi::Expression::tupleDims() const {
  TupleSizer visitor;
  this->accept(&visitor);
  return visitor.dims;
}

inline bi::Expression::operator bool() const {
  return true;
}

bool bi::Expression::operator<(Expression& o) {
  return *this <= o && o != *this;
}

bool bi::Expression::operator>(Expression& o) {
  return o <= *this && o != *this;
}

bool bi::Expression::operator>=(Expression& o) {
  return o <= *this;
}

bool bi::Expression::operator!=(Expression& o) {
  return !(*this == o);
}
