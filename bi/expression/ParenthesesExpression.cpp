/**
 * @file
 */
#include "bi/expression/ParenthesesExpression.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::ParenthesesExpression::ParenthesesExpression(Expression* single,
    shared_ptr<Location> loc) :
    Expression(loc),
    ExpressionUnary(single) {
  //
}

bi::ParenthesesExpression::~ParenthesesExpression() {
  //
}

bi::Expression* bi::ParenthesesExpression::strip() {
  return single->strip();
}

bi::Expression* bi::ParenthesesExpression::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::ParenthesesExpression::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ParenthesesExpression::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ParenthesesExpression::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this) || single->dispatchDefinitely(o);
}

bool bi::ParenthesesExpression::definitely(const BracesExpression& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(const BracketsExpression& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(const EmptyExpression& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(const List<Expression>& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(const FuncParameter& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(const FuncReference& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(Literal<bool>& o) {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(Literal<int64_t>& o) {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(const Literal<double>& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(Literal<const char*>& o) {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(const Member& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(const ParenthesesExpression& o) const {
  return single->definitely(*o.single);
}

bool bi::ParenthesesExpression::definitely(const Range& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(const Super& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(const This& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(const VarParameter& o) const {
  return (type->definitely(*o.type)) || single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(const VarReference& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this) || single->dispatchPossibly(o);
}

bool bi::ParenthesesExpression::possibly(const BracesExpression& o) const {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(const BracketsExpression& o) const {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(const EmptyExpression& o) const {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(const List<Expression>& o) const {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(const FuncParameter& o) const {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(const FuncReference& o) const {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(Literal<bool>& o) {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(Literal<int64_t>& o) {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(const Literal<double>& o) const {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(Literal<const char*>& o) {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(const Member& o) const {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(const ParenthesesExpression& o) const {
  return single->possibly(*o.single);
}

bool bi::ParenthesesExpression::possibly(const Range& o) const {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(const Super& o) const {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(const This& o) const {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(const VarParameter& o) const {
  return (type->possibly(*o.type)) || single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(const VarReference& o) const {
  return single->possibly(o);
}
