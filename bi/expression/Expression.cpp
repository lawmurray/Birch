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

bi::possibly bi::Expression::operator<=(Expression& o) {
  return o.dispatch(*this);
}

bi::possibly bi::Expression::operator==(Expression& o) {
  return *this <= o && o <= *this;
}

bi::possibly bi::Expression::le(BracesExpression& o) {
  return untrue;
}

bi::possibly bi::Expression::le(BracketsExpression& o) {
  return untrue;
}

bi::possibly bi::Expression::le(EmptyExpression& o) {
  return untrue;
}

bi::possibly bi::Expression::le(List<Expression>& o) {
  return untrue;
}

bi::possibly bi::Expression::le(FuncParameter& o) {
  return untrue;
}

bi::possibly bi::Expression::le(FuncReference& o) {
  return untrue;
}

bi::possibly bi::Expression::le(Index& o) {
  return untrue;
}

bi::possibly bi::Expression::le(Literal<bool>& o) {
  return untrue;
}

bi::possibly bi::Expression::le(Literal<int64_t>& o) {
  return untrue;
}

bi::possibly bi::Expression::le(Literal<double>& o) {
  return untrue;
}

bi::possibly bi::Expression::le(Literal<const char*>& o) {
  return untrue;
}

bi::possibly bi::Expression::le(ParenthesesExpression& o) {
  return untrue;
}

bi::possibly bi::Expression::le(Range& o) {
  return untrue;
}

bi::possibly bi::Expression::le(This& o) {
  return untrue;
}

bi::possibly bi::Expression::le(Member& o) {
  return untrue;
}

bi::possibly bi::Expression::le(RandomInit& o) {
  return untrue;
}

bi::possibly bi::Expression::le(VarParameter& o) {
  return untrue;
}

bi::possibly bi::Expression::le(VarReference& o) {
  return untrue;
}
