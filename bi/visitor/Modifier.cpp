/**
 * @file
 */
#include "bi/visitor/Modifier.hpp"

bi::Modifier::~Modifier() {
  //
}

void bi::Modifier::modify(File* o) {
  o->root = o->root->accept(this);
}

bi::Expression* bi::Modifier::modify(EmptyExpression* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(List<Expression>* o) {
  o->head = o->head->accept(this);
  o->tail = o->tail->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Literal<bool>* o) {
  o->type = o->type->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Literal<int64_t>* o) {
  o->type = o->type->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Literal<double>* o) {
  o->type = o->type->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Literal<const char*>* o) {
  o->type = o->type->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Parentheses* o) {
  o->single = o->single->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Brackets* o) {
  o->single = o->single->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Binary* o) {
  o->left = o->left->accept(this);
  o->right = o->right->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Call* o) {
  o->single = o->single->accept(this);
  o->args = o->args->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(BinaryCall* o) {
  o->single = o->single->accept(this);
  o->args = o->args->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(UnaryCall* o) {
  o->single = o->single->accept(this);
  o->args = o->args->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Slice* o) {
  o->single = o->single->accept(this);
  o->brackets = o->brackets->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(LambdaFunction* o) {
  o->parens = o->parens->accept(this);
  o->returnType = o->returnType->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Span* o) {
  o->single = o->single->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Index* o) {
  o->single = o->single->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Range* o) {
  o->left = o->left->accept(this);
  o->right = o->right->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Member* o) {
  o->left = o->left->accept(this);
  o->right = o->right->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Super* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(This* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(Parameter* o) {
  o->type = o->type->accept(this);
  o->value = o->value->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(MemberParameter* o) {
  o->type = o->type->accept(this);
  o->value = o->value->accept(this);
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
  o->left = o->left->accept(this);
  o->right = o->right->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(EmptyStatement* o) {
  return o;
}

bi::Statement* bi::Modifier::modify(List<Statement>* o) {
  o->head = o->head->accept(this);
  o->tail = o->tail->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(GlobalVariable* o) {
  o->type = o->type->accept(this);
  o->parens = o->parens->accept(this);
  o->value = o->value->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(LocalVariable* o) {
  o->type = o->type->accept(this);
  o->parens = o->parens->accept(this);
  o->value = o->value->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(MemberVariable* o) {
  o->type = o->type->accept(this);
  o->parens = o->parens->accept(this);
  o->value = o->value->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Function* o) {
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Coroutine* o) {
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Program* o) {
  o->params = o->params->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(MemberFunction* o) {
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(MemberCoroutine* o) {
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(BinaryOperator* o) {
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(UnaryOperator* o) {
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(AssignmentOperator* o) {
  o->single = o->single->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(ConversionOperator* o) {
  o->returnType = o->returnType->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Class* o) {
  o->parens = o->parens->accept(this);
  o->base = o->base->accept(this);
  o->baseParens = o->baseParens->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Alias* o) {
  o->base = o->base->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Basic* o) {
  return o;
}

bi::Statement* bi::Modifier::modify(Import* o) {
  return o;
}

bi::Statement* bi::Modifier::modify(ExpressionStatement* o) {
  o->single = o->single->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(If* o) {
  o->cond = o->cond->accept(this);
  o->braces = o->braces->accept(this);
  o->falseBraces = o->falseBraces->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(For* o) {
  o->index = o->index->accept(this);
  o->from = o->from->accept(this);
  o->to = o->to->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(While* o) {
  o->cond = o->cond->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Assert* o) {
  o->cond = o->cond->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Return* o) {
  o->single = o->single->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Yield* o) {
  o->single = o->single->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Raw* o) {
  return o;
}

bi::Type* bi::Modifier::modify(EmptyType* o) {
  return o;
}

bi::Type* bi::Modifier::modify(ListType* o) {
  o->head = o->head->accept(this);
  o->tail = o->tail->accept(this);
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
  o->single = o->single->accept(this);
  o->brackets = o->brackets->accept(this);
  return o;
}

bi::Type* bi::Modifier::modify(ParenthesesType* o) {
  o->single = o->single->accept(this);
  return o;
}

bi::Type* bi::Modifier::modify(BinaryType* o) {
  o->left = o->left->accept(this);
  o->right = o->right->accept(this);
  return o;
}

bi::Type* bi::Modifier::modify(FunctionType* o) {
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  return o;
}

bi::Type* bi::Modifier::modify(OverloadedType* o) {
  return o;
}

bi::Type* bi::Modifier::modify(FiberType* o) {
  o->single = o->single->accept(this);
  return o;
}

bi::Type* bi::Modifier::modify(OptionalType* o) {
  o->single = o->single->accept(this);
  return o;
}
