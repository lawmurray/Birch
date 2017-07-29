/**
 * @file
 */
#include "bi/visitor/Modifier.hpp"

bi::Modifier::~Modifier() {
  //
}

void bi::Modifier::modify(File* o) {
  o->root = o->root.release()->accept(this);
}

bi::Expression* bi::Modifier::modify(EmptyExpression* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(List<Expression>* o) {
  o->head = o->head.release()->accept(this);
  o->tail = o->tail.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Literal<bool>* o) {
  o->type = o->type.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Literal<int64_t>* o) {
  o->type = o->type.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Literal<double>* o) {
  o->type = o->type.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Literal<const char*>* o) {
  o->type = o->type.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Parentheses* o) {
  o->single = o->single.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Brackets* o) {
  o->single = o->single.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Call* o) {
  o->single = o->single.release()->accept(this);
  o->parens = o->parens.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(OverloadedCall<Function>* o) {
  o->single = o->single.release()->accept(this);
  o->parens = o->parens.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(OverloadedCall<Coroutine>* o) {
  o->single = o->single.release()->accept(this);
  o->parens = o->parens.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(OverloadedCall<MemberFunction>* o) {
  o->single = o->single.release()->accept(this);
  o->parens = o->parens.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(OverloadedCall<MemberCoroutine>* o) {
  o->single = o->single.release()->accept(this);
  o->parens = o->parens.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(OverloadedCall<BinaryOperator>* o) {
  o->left = o->left.release()->accept(this);
  o->right = o->right.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(OverloadedCall<UnaryOperator>* o) {
  o->single = o->single.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Slice* o) {
  o->single = o->single.release()->accept(this);
  o->brackets = o->brackets.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(LambdaFunction* o) {
  o->parens = o->parens.release()->accept(this);
  o->returnType = o->returnType.release()->accept(this);
  o->braces = o->braces.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Span* o) {
  o->single = o->single.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Index* o) {
  o->single = o->single.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Range* o) {
  o->left = o->left.release()->accept(this);
  o->right = o->right.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Member* o) {
  o->left = o->left.release()->accept(this);
  o->right = o->right.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Super* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(This* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(Parameter* o) {
  o->type = o->type.release()->accept(this);
  o->value = o->value.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(MemberParameter* o) {
  o->type = o->type.release()->accept(this);
  o->value = o->value.release()->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Identifier<Unknown>* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(Identifier<Parameter>* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(Identifier<MemberParameter>* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(Identifier<GlobalVariable>* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(Identifier<LocalVariable>* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(Identifier<MemberVariable>* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(OverloadedIdentifier<Function>* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(OverloadedIdentifier<Coroutine>* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(OverloadedIdentifier<MemberCoroutine>* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(OverloadedIdentifier<MemberFunction>* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(OverloadedIdentifier<BinaryOperator>* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(OverloadedIdentifier<UnaryOperator>* o) {
  return o;
}

bi::Statement* bi::Modifier::modify(Assignment* o) {
  o->left = o->left.release()->accept(this);
  o->right = o->right.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(EmptyStatement* o) {
  return o;
}

bi::Statement* bi::Modifier::modify(List<Statement>* o) {
  o->head = o->head.release()->accept(this);
  o->tail = o->tail.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(GlobalVariable* o) {
  o->type = o->type.release()->accept(this);
  o->parens = o->parens.release()->accept(this);
  o->value = o->value.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(LocalVariable* o) {
  o->type = o->type.release()->accept(this);
  o->parens = o->parens.release()->accept(this);
  o->value = o->value.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(MemberVariable* o) {
  o->type = o->type.release()->accept(this);
  o->parens = o->parens.release()->accept(this);
  o->value = o->value.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Function* o) {
  o->parens = o->parens.release()->accept(this);
  o->returnType = o->returnType.release()->accept(this);
  o->braces = o->braces.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Coroutine* o) {
  o->parens = o->parens.release()->accept(this);
  o->returnType = o->returnType.release()->accept(this);
  o->braces = o->braces.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Program* o) {
  o->parens = o->parens.release()->accept(this);
  o->braces = o->braces.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(MemberFunction* o) {
  o->parens = o->parens.release()->accept(this);
  o->returnType = o->returnType.release()->accept(this);
  o->braces = o->braces.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(MemberCoroutine* o) {
  o->parens = o->parens.release()->accept(this);
  o->returnType = o->returnType.release()->accept(this);
  o->braces = o->braces.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(BinaryOperator* o) {
  o->left = o->left.release()->accept(this);
  o->right = o->right.release()->accept(this);
  o->returnType = o->returnType.release()->accept(this);
  o->braces = o->braces.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(UnaryOperator* o) {
  o->single = o->single.release()->accept(this);
  o->returnType = o->returnType.release()->accept(this);
  o->braces = o->braces.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(AssignmentOperator* o) {
  o->single = o->single.release()->accept(this);
  o->braces = o->braces.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(ConversionOperator* o) {
  o->returnType = o->returnType.release()->accept(this);
  o->braces = o->braces.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Class* o) {
  o->parens = o->parens.release()->accept(this);
  o->base = o->base.release()->accept(this);
  o->baseParens = o->baseParens.release()->accept(this);
  o->braces = o->braces.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Alias* o) {
  o->base = o->base.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Basic* o) {
  return o;
}

bi::Statement* bi::Modifier::modify(Import* o) {
  return o;
}

bi::Statement* bi::Modifier::modify(ExpressionStatement* o) {
  o->single = o->single.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(If* o) {
  o->cond = o->cond.release()->accept(this);
  o->braces = o->braces.release()->accept(this);
  o->falseBraces = o->falseBraces.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(For* o) {
  o->index = o->index.release()->accept(this);
  o->from = o->from.release()->accept(this);
  o->to = o->to.release()->accept(this);
  o->braces = o->braces.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(While* o) {
  o->cond = o->cond.release()->accept(this);
  o->braces = o->braces.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Assert* o) {
  o->cond = o->cond.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Return* o) {
  o->single = o->single.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Yield* o) {
  o->single = o->single.release()->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Raw* o) {
  return o;
}

bi::Type* bi::Modifier::modify(EmptyType* o) {
  return o;
}

bi::Type* bi::Modifier::modify(List<Type>* o) {
  o->head = o->head.release()->accept(this);
  o->tail = o->tail.release()->accept(this);
  return o;
}

bi::Type* bi::Modifier::modify(IdentifierType* o) {
  return o;
}

bi::Type* bi::Modifier::modify(ClassType* o) {
  return o;
}

bi::Type* bi::Modifier::modify(AliasType* o) {
  return o;
}

bi::Type* bi::Modifier::modify(BasicType* o) {
  return o;
}

bi::Type* bi::Modifier::modify(ArrayType* o) {
  o->single = o->single.release()->accept(this);
  o->brackets = o->brackets.release()->accept(this);
  return o;
}

bi::Type* bi::Modifier::modify(ParenthesesType* o) {
  o->single = o->single.release()->accept(this);
  return o;
}

bi::Type* bi::Modifier::modify(FunctionType* o) {
  o->parens = o->parens.release()->accept(this);
  o->returnType = o->returnType.release()->accept(this);
  return o;
}

bi::Type* bi::Modifier::modify(FiberType* o) {
  o->returnType = o->returnType.release()->accept(this);
  return o;
}

