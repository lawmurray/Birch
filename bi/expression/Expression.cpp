/**
 * @file
 */
#include "bi/expression/Expression.hpp"

#include "bi/type/EmptyType.hpp"
#include "bi/visitor/TupleSizer.hpp"
#include "bi/visitor/Gatherer.hpp"

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

bool bi::Expression::hasAssignable() const {
  Gatherer<VarParameter> gatherer;
  accept(&gatherer);
  for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
    if ((*iter)->type->assignable) {
      return true;
    }
  }
  return false;
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

bi::Iterator<bi::Expression> bi::Expression::begin() const {
  return bi::Iterator<Expression>(this);
}

bi::Iterator<bi::Expression> bi::Expression::end() const {
  return bi::Iterator<Expression>(nullptr);
}

bool bi::Expression::definitely(const Expression& o) const {
  return o.dispatchDefinitely(*this);
}

bool bi::Expression::definitely(const BracesExpression& o) const {
  return false;
}

bool bi::Expression::definitely(const BracketsExpression& o) const {
  return false;
}

bool bi::Expression::definitely(const EmptyExpression& o) const {
  return false;
}

bool bi::Expression::definitely(const List<Expression>& o) const {
  return false;
}

bool bi::Expression::definitely(const FuncParameter& o) const {
  return false;
}

bool bi::Expression::definitely(const FuncReference& o) const {
  return false;
}

bool bi::Expression::definitely(const Index& o) const {
  return false;
}

bool bi::Expression::definitely(const Literal<bool>& o) const {
  return false;
}

bool bi::Expression::definitely(Literal<int64_t>& o) {
  return false;
}

bool bi::Expression::definitely(const Literal<double>& o) const {
  return false;
}

bool bi::Expression::definitely(Literal<const char*>& o) {
  return false;
}

bool bi::Expression::definitely(const ParenthesesExpression& o) const {
  return false;
}

bool bi::Expression::definitely(const Range& o) const {
  return false;
}

bool bi::Expression::definitely(const This& o) const {
  return false;
}

bool bi::Expression::definitely(const Member& o) const {
  return false;
}

bool bi::Expression::definitely(const VarParameter& o) const {
  return false;
}

bool bi::Expression::definitely(const VarReference& o) const {
  return false;
}

bool bi::Expression::possibly(const Expression& o) const {
  return o.dispatchPossibly(*this);
}

bool bi::Expression::possibly(const BracesExpression& o) const {
  return false;
}

bool bi::Expression::possibly(const BracketsExpression& o) const {
  return false;
}

bool bi::Expression::possibly(const EmptyExpression& o) const {
  return false;
}

bool bi::Expression::possibly(const List<Expression>& o) const {
  return false;
}

bool bi::Expression::possibly(const FuncParameter& o) const {
  return false;
}

bool bi::Expression::possibly(const FuncReference& o) const {
  return false;
}

bool bi::Expression::possibly(const Index& o) const {
  return false;
}

bool bi::Expression::possibly(const Literal<bool>& o) const {
  return false;
}

bool bi::Expression::possibly(Literal<int64_t>& o) {
  return false;
}

bool bi::Expression::possibly(const Literal<double>& o) const {
  return false;
}

bool bi::Expression::possibly(Literal<const char*>& o) {
  return false;
}

bool bi::Expression::possibly(const ParenthesesExpression& o) const {
  return false;
}

bool bi::Expression::possibly(const Range& o) const {
  return false;
}

bool bi::Expression::possibly(const This& o) const {
  return false;
}

bool bi::Expression::possibly(const Member& o) const {
  return false;
}

bool bi::Expression::possibly(const VarParameter& o) const {
  return false;
}

bool bi::Expression::possibly(const VarReference& o) const {
  return false;
}

bool bi::Expression::operator==(Expression& o) {
  return definitely(o) && o.definitely(*this);
}

bool bi::Expression::operator!=(Expression& o) {
  return !(*this == o);
}
