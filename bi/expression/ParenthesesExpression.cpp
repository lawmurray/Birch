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

bool bi::ParenthesesExpression::dispatchDefinitely(Expression& o) {
  return o.definitely(*this) || single->dispatchDefinitely(o);
}

bool bi::ParenthesesExpression::definitely(BracesExpression& o) {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(BracketsExpression& o) {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(EmptyExpression& o) {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(List<Expression>& o) {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(FuncParameter& o) {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(FuncReference& o) {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(Literal<unsigned char>& o) {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(Literal<int64_t>& o) {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(Literal<double>& o) {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(Literal<const char*>& o) {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(Member& o) {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(ParenthesesExpression& o) {
  return single->definitely(*o.single);
}

bool bi::ParenthesesExpression::definitely(RandomInit& o) {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(Range& o) {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(This& o) {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(VarParameter& o) {
  return (type->definitely(*o.type) && o.capture(this)) || single->definitely(o);
}

bool bi::ParenthesesExpression::definitely(VarReference& o) {
  return single->definitely(o);
}

bool bi::ParenthesesExpression::dispatchPossibly(Expression& o) {
  return o.possibly(*this) || single->dispatchPossibly(o);
}

bool bi::ParenthesesExpression::possibly(BracesExpression& o) {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(BracketsExpression& o) {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(EmptyExpression& o) {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(List<Expression>& o) {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(FuncParameter& o) {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(FuncReference& o) {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(Literal<unsigned char>& o) {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(Literal<int64_t>& o) {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(Literal<double>& o) {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(Literal<const char*>& o) {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(Member& o) {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(ParenthesesExpression& o) {
  return single->possibly(*o.single);
}

bool bi::ParenthesesExpression::possibly(RandomInit& o) {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(Range& o) {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(This& o) {
  return single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(VarParameter& o) {
  return (type->possibly(*o.type) && o.capture(this)) || single->possibly(o);
}

bool bi::ParenthesesExpression::possibly(VarReference& o) {
  return single->possibly(o);
}
