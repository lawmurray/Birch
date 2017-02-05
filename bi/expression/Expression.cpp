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

bool bi::Expression::definitely(Expression& o) {
  return o.dispatchDefinitely(*this);
}

bool bi::Expression::definitely(BracesExpression& o) {
  return false;
}

bool bi::Expression::definitely(BracketsExpression& o) {
  return false;
}

bool bi::Expression::definitely(EmptyExpression& o) {
  return false;
}

bool bi::Expression::definitely(List<Expression>& o) {
  return false;
}

bool bi::Expression::definitely(FuncParameter& o) {
  return false;
}

bool bi::Expression::definitely(FuncReference& o) {
  return false;
}

bool bi::Expression::definitely(Index& o) {
  return false;
}

bool bi::Expression::definitely(Literal<bool>& o) {
  return false;
}

bool bi::Expression::definitely(Literal<int64_t>& o) {
  return false;
}

bool bi::Expression::definitely(Literal<double>& o) {
  return false;
}

bool bi::Expression::definitely(Literal<const char*>& o) {
  return false;
}

bool bi::Expression::definitely(ParenthesesExpression& o) {
  return false;
}

bool bi::Expression::definitely(Range& o) {
  return false;
}

bool bi::Expression::definitely(This& o) {
  return false;
}

bool bi::Expression::definitely(Member& o) {
  return false;
}

bool bi::Expression::definitely(RandomInit& o) {
  return false;
}

bool bi::Expression::definitely(VarParameter& o) {
  return false;
}

bool bi::Expression::definitely(VarReference& o) {
  return false;
}

bool bi::Expression::possibly(Expression& o) {
  return o.dispatchPossibly(*this);
}

bool bi::Expression::possibly(BracesExpression& o) {
  return false;
}

bool bi::Expression::possibly(BracketsExpression& o) {
  return false;
}

bool bi::Expression::possibly(EmptyExpression& o) {
  return false;
}

bool bi::Expression::possibly(List<Expression>& o) {
  return false;
}

bool bi::Expression::possibly(FuncParameter& o) {
  return false;
}

bool bi::Expression::possibly(FuncReference& o) {
  return false;
}

bool bi::Expression::possibly(Index& o) {
  return false;
}

bool bi::Expression::possibly(Literal<bool>& o) {
  return false;
}

bool bi::Expression::possibly(Literal<int64_t>& o) {
  return false;
}

bool bi::Expression::possibly(Literal<double>& o) {
  return false;
}

bool bi::Expression::possibly(Literal<const char*>& o) {
  return false;
}

bool bi::Expression::possibly(ParenthesesExpression& o) {
  return false;
}

bool bi::Expression::possibly(Range& o) {
  return false;
}

bool bi::Expression::possibly(This& o) {
  return false;
}

bool bi::Expression::possibly(Member& o) {
  return false;
}

bool bi::Expression::possibly(RandomInit& o) {
  return false;
}

bool bi::Expression::possibly(VarParameter& o) {
  return false;
}

bool bi::Expression::possibly(VarReference& o) {
  return false;
}
