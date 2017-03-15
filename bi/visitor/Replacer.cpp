/**
 * @file
 */
#include "bi/visitor/Replacer.hpp"

bi::Replacer::Replacer(Expression* find, Expression* replace) :
    find(find),
    replace(replace) {
  //
}

bi::Replacer::~Replacer() {
  //
}

bi::Expression* bi::Replacer::modify(EmptyExpression* o) {
  Modifier::modify(o);
  return (o == find) ? replace : o;
}

bi::Expression* bi::Replacer::modify(BooleanLiteral* o) {
  Modifier::modify(o);
  return (o == find) ? replace : o;
}

bi::Expression* bi::Replacer::modify(IntegerLiteral* o) {
  Modifier::modify(o);
  return (o == find) ? replace : o;
}

bi::Expression* bi::Replacer::modify(RealLiteral* o) {
  Modifier::modify(o);
  return (o == find) ? replace : o;
}

bi::Expression* bi::Replacer::modify(StringLiteral* o) {
  Modifier::modify(o);
  return (o == find) ? replace : o;
}

bi::Expression* bi::Replacer::modify(ExpressionList* o) {
  Modifier::modify(o);
  return (o == find) ? replace : o;
}

bi::Expression* bi::Replacer::modify(ParenthesesExpression* o) {
  Modifier::modify(o);
  return (o == find) ? replace : o;
}

bi::Expression* bi::Replacer::modify(BracesExpression* o) {
  Modifier::modify(o);
  return (o == find) ? replace : o;
}

bi::Expression* bi::Replacer::modify(BracketsExpression* o) {
  Modifier::modify(o);
  return (o == find) ? replace : o;
}

bi::Expression* bi::Replacer::modify(Index* o) {
  Modifier::modify(o);
  return (o == find) ? replace : o;
}

bi::Expression* bi::Replacer::modify(Range* o) {
  Modifier::modify(o);
  return (o == find) ? replace : o;
}

bi::Expression* bi::Replacer::modify(Member* o) {
  Modifier::modify(o);
  return (o == find) ? replace : o;
}

bi::Expression* bi::Replacer::modify(This* o) {
  Modifier::modify(o);
  return (o == find) ? replace : o;
}

bi::Expression* bi::Replacer::modify(LambdaInit* o) {
  Modifier::modify(o);
  return (o == find) ? replace : o;
}

bi::Expression* bi::Replacer::modify(RandomInit* o) {
  Modifier::modify(o);
  return (o == find) ? replace : o;
}

bi::Expression* bi::Replacer::modify(VarReference* o) {
  Modifier::modify(o);
  return (o == find) ? replace : o;
}

bi::Expression* bi::Replacer::modify(FuncReference* o) {
  Modifier::modify(o);
  return (o == find) ? replace : o;
}

bi::Expression* bi::Replacer::modify(VarParameter* o) {
  Modifier::modify(o);
  return (o == find) ? replace : o;
}

bi::Expression* bi::Replacer::modify(FuncParameter* o) {
  Modifier::modify(o);
  return (o == find) ? replace : o;
}
