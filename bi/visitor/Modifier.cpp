/**
 * @file
 */
#include "bi/visitor/Modifier.hpp"

bi::Modifier::~Modifier() {
  //
}

bi::Package* bi::Modifier::modify(Package* o) {
  for (auto file : o->files) {
    file = file->accept(this);
  }
  return o;
}

bi::File* bi::Modifier::modify(File* o) {
  o->root = o->root->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(EmptyExpression* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(ExpressionList* o) {
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

bi::Expression* bi::Modifier::modify(Sequence* o) {
  o->single = o->single->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Binary* o) {
  o->left = o->left->accept(this);
  o->right = o->right->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Cast* o) {
  o->returnType = o->returnType->accept(this);
  o->single = o->single->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Call<Unknown>* o) {
  o->single = o->single->accept(this);
  o->args = o->args->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Call<Function>* o) {
  o->single = o->single->accept(this);
  o->args = o->args->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Call<MemberFunction>* o) {
  o->single = o->single->accept(this);
  o->args = o->args->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Call<Fiber>* o) {
  o->single = o->single->accept(this);
  o->args = o->args->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Call<MemberFiber>* o) {
  o->single = o->single->accept(this);
  o->args = o->args->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Call<Parameter>* o) {
  o->single = o->single->accept(this);
  o->args = o->args->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Call<LocalVariable>* o) {
  o->single = o->single->accept(this);
  o->args = o->args->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Call<MemberVariable>* o) {
  o->single = o->single->accept(this);
  o->args = o->args->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Call<GlobalVariable>* o) {
  o->single = o->single->accept(this);
  o->args = o->args->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Call<BinaryOperator>* o) {
  o->single = o->single->accept(this);
  o->args = o->args->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Call<UnaryOperator>* o) {
  o->single = o->single->accept(this);
  o->args = o->args->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Assign* o) {
  o->left = o->left->accept(this);
  o->right = o->right->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Slice* o) {
  o->single = o->single->accept(this);
  o->brackets = o->brackets->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Query* o) {
  o->single = o->single->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Get* o) {
  o->single = o->single->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(LambdaFunction* o) {
  o->params = o->params->accept(this);
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

bi::Expression* bi::Modifier::modify(Global* o) {
  o->single = o->single->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Super* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(This* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(Nil* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(Parameter* o) {
  o->type = o->type->accept(this);
  o->value = o->value->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Generic* o) {
  o->type = o->type->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Identifier<Unknown>* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(Identifier<Parameter>* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(Identifier<GlobalVariable>* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(Identifier<MemberVariable>* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(Identifier<LocalVariable>* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(Identifier<ForVariable>* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(Identifier<ParallelVariable>* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(OverloadedIdentifier<Unknown>* o) {
  o->typeArgs = o->typeArgs->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(OverloadedIdentifier<Function>* o) {
  o->typeArgs = o->typeArgs->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(OverloadedIdentifier<Fiber>* o) {
  o->typeArgs = o->typeArgs->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(OverloadedIdentifier<MemberFiber>* o) {
  o->typeArgs = o->typeArgs->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(OverloadedIdentifier<MemberFunction>* o) {
  o->typeArgs = o->typeArgs->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(OverloadedIdentifier<BinaryOperator>* o) {
  o->typeArgs = o->typeArgs->accept(this);
  return o;
}

bi::Expression* bi::Modifier::modify(OverloadedIdentifier<UnaryOperator>* o) {
  o->typeArgs = o->typeArgs->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Assume* o) {
  o->left = o->left->accept(this);
  o->right = o->right->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(EmptyStatement* o) {
  return o;
}

bi::Statement* bi::Modifier::modify(Braces* o) {
  o->single = o->single->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(StatementList* o) {
  o->head = o->head->accept(this);
  o->tail = o->tail->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(GlobalVariable* o) {
  o->type = o->type->accept(this);
  o->brackets = o->brackets->accept(this);
  o->args = o->args->accept(this);
  o->value = o->value->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(MemberVariable* o) {
  o->type = o->type->accept(this);
  o->brackets = o->brackets->accept(this);
  o->args = o->args->accept(this);
  o->value = o->value->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(LocalVariable* o) {
  o->type = o->type->accept(this);
  o->brackets = o->brackets->accept(this);
  o->args = o->args->accept(this);
  o->value = o->value->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(ForVariable* o) {
  o->type = o->type->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(ParallelVariable* o) {
  o->type = o->type->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Function* o) {
  o->typeParams = o->typeParams->accept(this);
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Fiber* o) {
  o->typeParams = o->typeParams->accept(this);
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

bi::Statement* bi::Modifier::modify(MemberFiber* o) {
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
  o->typeParams = o->typeParams->accept(this);
  o->params = o->params->accept(this);
  o->base = o->base->accept(this);
  o->args = o->args->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Basic* o) {
  o->base = o->base->accept(this);
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

bi::Statement* bi::Modifier::modify(Parallel* o) {
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

bi::Statement* bi::Modifier::modify(DoWhile* o) {
  o->braces = o->braces->accept(this);
  o->cond = o->cond->accept(this);
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

bi::Statement* bi::Modifier::modify(Instantiated<Type>* o) {
  o->single = o->single->accept(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Instantiated<Expression>* o) {
  o->single = o->single->accept(this);
  return o;
}

bi::Type* bi::Modifier::modify(EmptyType* o) {
  return o;
}

bi::Type* bi::Modifier::modify(TypeList* o) {
  o->head = o->head->accept(this);
  o->tail = o->tail->accept(this);
  return o;
}

bi::Type* bi::Modifier::modify(UnknownType* o) {
  o->typeArgs = o->typeArgs->accept(this);
  return o;
}

bi::Type* bi::Modifier::modify(ClassType* o) {
  o->typeArgs = o->typeArgs->accept(this);
  return o;
}

bi::Type* bi::Modifier::modify(BasicType* o) {
  return o;
}

bi::Type* bi::Modifier::modify(GenericType* o) {
  return o;
}

bi::Type* bi::Modifier::modify(MemberType* o) {
  o->left = o->left->accept(this);
  o->right = o->right->accept(this);
  return o;
}

bi::Type* bi::Modifier::modify(ArrayType* o) {
  o->single = o->single->accept(this);
  return o;
}

bi::Type* bi::Modifier::modify(TupleType* o) {
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

bi::Type* bi::Modifier::modify(FiberType* o) {
  o->yieldType = o->yieldType->accept(this);
  return o;
}

bi::Type* bi::Modifier::modify(OptionalType* o) {
  o->single = o->single->accept(this);
  return o;
}

bi::Type* bi::Modifier::modify(NilType* o) {
  return o;
}
