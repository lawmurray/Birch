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

bool bi::Expression::isEmpty() const {
  return false;
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

bool bi::Expression::operator<=(Expression& o) {
  return o.dispatch(*this);
}

bool bi::Expression::operator==(Expression& o) {
  return *this <= o && o <= *this;
}

bool bi::Expression::le(BracesExpression& o) {
  return false;
}

bool bi::Expression::le(BracketsExpression& o) {
  return false;
}

bool bi::Expression::le(EmptyExpression& o) {
  return false;
}

bool bi::Expression::le(List<Expression>& o) {
  return false;
}

bool bi::Expression::le(FuncParameter& o) {
  return false;
}

bool bi::Expression::le(FuncReference& o) {
  return false;
}

bool bi::Expression::le(Literal<bool>& o) {
  return false;
}

bool bi::Expression::le(Literal<int64_t>& o) {
  return false;
}

bool bi::Expression::le(Literal<double>& o) {
  return false;
}

bool bi::Expression::le(Literal<std::string>& o) {
  return false;
}

bool bi::Expression::le(ParenthesesExpression& o) {
  return false;
}

bool bi::Expression::le(RandomParameter& o) {
  return false;
}

bool bi::Expression::le(RandomReference& o) {
  return false;
}

bool bi::Expression::le(RandomRight& o) {
  return false;
}

bool bi::Expression::le(Range& o) {
  return false;
}

bool bi::Expression::le(This& o) {
  return false;
}

bool bi::Expression::le(Member& o) {
  return false;
}

bool bi::Expression::le(VarParameter& o) {
  return false;
}

bool bi::Expression::le(VarReference& o) {
  return false;
}
