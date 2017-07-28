/**
 * @file
 */
#include "bi/expression/Parentheses.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

bi::Parentheses::Parentheses(Expression* single,
    shared_ptr<Location> loc) :
    Expression(loc),
    Unary<Expression>(single) {
  //
}

bi::Parentheses::~Parentheses() {
  //
}

bi::Expression* bi::Parentheses::strip() {
  return single->strip();
}

bi::Iterator<bi::Expression> bi::Parentheses::begin() const {
  return single->begin();
}

bi::Iterator<bi::Expression> bi::Parentheses::end() const {
  return single->end();
}

bi::Expression* bi::Parentheses::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Parentheses::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Parentheses::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Parentheses::dispatchDefinitely(
    const Expression& o) const {
  return o.definitely(*this) || single->dispatchDefinitely(o);
}

bool bi::Parentheses::definitely(const OverloadedCall<BinaryOperator>& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(const Call& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(const EmptyExpression& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(
    const Identifier<Parameter>& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(
    const Identifier<GlobalVariable>& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(
    const Identifier<LocalVariable>& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(
    const Identifier<MemberVariable>& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(
    const OverloadedIdentifier<Function>& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(
    const OverloadedIdentifier<Coroutine>& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(
    const OverloadedIdentifier<MemberFunction>& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(
    const OverloadedIdentifier<MemberCoroutine>& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(const Index& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(const LambdaFunction& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(const List<Expression>& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(const Literal<bool>& o) {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(const Literal<int64_t>& o) {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(const Literal<double>& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(const Literal<const char*>& o) {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(const Member& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(const Parameter& o) const {
  return (type->definitely(*o.type)) || single->definitely(o);
}

bool bi::Parentheses::definitely(
    const Parentheses& o) const {
  return single->definitely(*o.single);
}

bool bi::Parentheses::definitely(const Range& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(
    const Slice& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(const Span& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(const Super& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(const This& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::definitely(const OverloadedCall<UnaryOperator>& o) const {
  return single->definitely(o);
}

bool bi::Parentheses::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this) || single->dispatchPossibly(o);
}

bool bi::Parentheses::possibly(const OverloadedCall<BinaryOperator>& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(const Call& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(const EmptyExpression& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(
    const Identifier<Parameter>& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(
    const Identifier<GlobalVariable>& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(
    const Identifier<LocalVariable>& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(
    const Identifier<MemberVariable>& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(
    const OverloadedIdentifier<Function>& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(
    const OverloadedIdentifier<Coroutine>& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(
    const OverloadedIdentifier<MemberFunction>& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(
    const OverloadedIdentifier<MemberCoroutine>& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(const Index& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(const LambdaFunction& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(const List<Expression>& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(const Literal<bool>& o) {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(const Literal<int64_t>& o) {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(const Literal<double>& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(const Literal<const char*>& o) {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(const Member& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(const Parameter& o) const {
  return (type->possibly(*o.type)) || single->possibly(o);
}

bool bi::Parentheses::possibly(
    const Parentheses& o) const {
  return single->possibly(*o.single);
}

bool bi::Parentheses::possibly(const Range& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(const Slice& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(const Span& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(const Super& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(const This& o) const {
  return single->possibly(o);
}

bool bi::Parentheses::possibly(const OverloadedCall<UnaryOperator>& o) const {
  return single->possibly(o);
}
