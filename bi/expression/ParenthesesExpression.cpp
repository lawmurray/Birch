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

bool bi::ParenthesesExpression::dispatch(Expression& o) {
  return o.le(*this) || single->dispatch(o);
}

bool bi::ParenthesesExpression::le(BracesExpression& o) {
  return *single <= o;
}

bool bi::ParenthesesExpression::le(BracketsExpression& o) {
  return *single <= o;
}

bool bi::ParenthesesExpression::le(EmptyExpression& o) {
  return *single <= o;
}

bool bi::ParenthesesExpression::le(List<Expression>& o) {
  return *single <= o;
}

bool bi::ParenthesesExpression::le(FuncParameter& o) {
  return *single <= o;
}

bool bi::ParenthesesExpression::le(FuncReference& o) {
  return *single <= o;
}

bool bi::ParenthesesExpression::le(Literal<bool>& o) {
  return *single <= o;
}

bool bi::ParenthesesExpression::le(Literal<int64_t>& o) {
  return *single <= o;
}

bool bi::ParenthesesExpression::le(Literal<double>& o) {
  return *single <= o;
}

bool bi::ParenthesesExpression::le(Literal<std::string>& o) {
  return *single <= o;
}

bool bi::ParenthesesExpression::le(Member& o) {
  return *single <= o;
}

bool bi::ParenthesesExpression::le(ParenthesesExpression& o) {
  return *single <= *o.single;
}

bool bi::ParenthesesExpression::le(RandomParameter& o) {
  return *single <= o;
}

bool bi::ParenthesesExpression::le(RandomReference& o) {
  return *single <= o;
}

bool bi::ParenthesesExpression::le(Range& o) {
  return *single <= o;
}

bool bi::ParenthesesExpression::le(This& o) {
  return *single <= o;
}

bool bi::ParenthesesExpression::le(VarParameter& o) {
  return (*type <= *o.type && o.capture(this)) || *single <= o;
}

bool bi::ParenthesesExpression::le(VarReference& o) {
  return *single <= o;
}
