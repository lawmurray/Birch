/**
 * @file
 */
#include "src/visitor/Modifier.hpp"

birch::Modifier::~Modifier() {
  //
}

birch::Package* birch::Modifier::modify(Package* o) {
  for (auto file : o->files) {
    file = file->accept(this);
  }
  return o;
}

birch::File* birch::Modifier::modify(File* o) {
  o->root = o->root->accept(this);
  return o;
}

birch::Expression* birch::Modifier::modify(EmptyExpression* o) {
  return o;
}

birch::Expression* birch::Modifier::modify(ExpressionList* o) {
  o->head = o->head->accept(this);
  o->tail = o->tail->accept(this);
  return o;
}

birch::Expression* birch::Modifier::modify(Literal<bool>* o) {
  return o;
}

birch::Expression* birch::Modifier::modify(Literal<int64_t>* o) {
  return o;
}

birch::Expression* birch::Modifier::modify(Literal<double>* o) {
  return o;
}

birch::Expression* birch::Modifier::modify(Literal<const char*>* o) {
  return o;
}

birch::Expression* birch::Modifier::modify(Parentheses* o) {
  o->single = o->single->accept(this);
  return o;
}

birch::Expression* birch::Modifier::modify(Sequence* o) {
  o->single = o->single->accept(this);
  return o;
}

birch::Expression* birch::Modifier::modify(Cast* o) {
  o->returnType = o->returnType->accept(this);
  o->single = o->single->accept(this);
  return o;
}

birch::Expression* birch::Modifier::modify(Call* o) {
  o->single = o->single->accept(this);
  o->args = o->args->accept(this);
  return o;
}

birch::Expression* birch::Modifier::modify(BinaryCall* o) {
  o->left = o->left->accept(this);
  o->right = o->right->accept(this);
  return o;
}

birch::Expression* birch::Modifier::modify(UnaryCall* o) {
  o->single = o->single->accept(this);
  return o;
}

birch::Expression* birch::Modifier::modify(Assign* o) {
  o->left = o->left->accept(this);
  o->right = o->right->accept(this);
  return o;
}

birch::Expression* birch::Modifier::modify(Slice* o) {
  o->single = o->single->accept(this);
  o->brackets = o->brackets->accept(this);
  return o;
}

birch::Expression* birch::Modifier::modify(Query* o) {
  o->single = o->single->accept(this);
  return o;
}

birch::Expression* birch::Modifier::modify(Get* o) {
  o->single = o->single->accept(this);
  return o;
}

birch::Expression* birch::Modifier::modify(LambdaFunction* o) {
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

birch::Expression* birch::Modifier::modify(Span* o) {
  o->single = o->single->accept(this);
  return o;
}

birch::Expression* birch::Modifier::modify(Range* o) {
  o->left = o->left->accept(this);
  o->right = o->right->accept(this);
  return o;
}

birch::Expression* birch::Modifier::modify(Member* o) {
  o->left = o->left->accept(this);
  o->right = o->right->accept(this);
  return o;
}

birch::Expression* birch::Modifier::modify(Global* o) {
  o->single = o->single->accept(this);
  return o;
}

birch::Expression* birch::Modifier::modify(Super* o) {
  return o;
}

birch::Expression* birch::Modifier::modify(This* o) {
  return o;
}

birch::Expression* birch::Modifier::modify(Nil* o) {
  return o;
}

birch::Expression* birch::Modifier::modify(Parameter* o) {
  o->type = o->type->accept(this);
  o->value = o->value->accept(this);
  return o;
}

birch::Expression* birch::Modifier::modify(Generic* o) {
  o->type = o->type->accept(this);
  return o;
}

birch::Expression* birch::Modifier::modify(NamedExpression* o) {
  o->typeArgs = o->typeArgs->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(Assume* o) {
  o->left = o->left->accept(this);
  o->right = o->right->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(EmptyStatement* o) {
  return o;
}

birch::Statement* birch::Modifier::modify(Braces* o) {
  o->single = o->single->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(StatementList* o) {
  o->head = o->head->accept(this);
  o->tail = o->tail->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(GlobalVariable* o) {
  o->type = o->type->accept(this);
  o->brackets = o->brackets->accept(this);
  o->args = o->args->accept(this);
  o->value = o->value->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(MemberVariable* o) {
  o->type = o->type->accept(this);
  o->brackets = o->brackets->accept(this);
  o->args = o->args->accept(this);
  o->value = o->value->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(LocalVariable* o) {
  o->type = o->type->accept(this);
  o->brackets = o->brackets->accept(this);
  o->args = o->args->accept(this);
  o->value = o->value->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(Function* o) {
  o->typeParams = o->typeParams->accept(this);
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(Program* o) {
  o->params = o->params->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(MemberFunction* o) {
  o->typeParams = o->typeParams->accept(this);
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(BinaryOperator* o) {
  o->left = o->left->accept(this);
  o->right = o->right->accept(this);
  o->returnType = o->returnType->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(UnaryOperator* o) {
  o->single = o->single->accept(this);
  o->returnType = o->returnType->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(AssignmentOperator* o) {
  o->single = o->single->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(ConversionOperator* o) {
  o->returnType = o->returnType->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(SliceOperator* o) {
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(Class* o) {
  o->typeParams = o->typeParams->accept(this);
  o->params = o->params->accept(this);
  o->base = o->base->accept(this);
  o->args = o->args->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(Basic* o) {
  o->typeParams = o->typeParams->accept(this);
  o->base = o->base->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(ExpressionStatement* o) {
  o->single = o->single->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(If* o) {
  o->cond = o->cond->accept(this);
  o->braces = o->braces->accept(this);
  o->falseBraces = o->falseBraces->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(For* o) {
  o->index = o->index->accept(this);
  o->from = o->from->accept(this);
  o->to = o->to->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(Parallel* o) {
  o->index = o->index->accept(this);
  o->from = o->from->accept(this);
  o->to = o->to->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(While* o) {
  o->cond = o->cond->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(DoWhile* o) {
  o->braces = o->braces->accept(this);
  o->cond = o->cond->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(With* o) {
  o->single = o->single->accept(this);
  o->braces = o->braces->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(Block* o) {
  o->braces = o->braces->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(Assert* o) {
  o->cond = o->cond->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(Return* o) {
  o->single = o->single->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(Factor* o) {
  o->single = o->single->accept(this);
  return o;
}

birch::Statement* birch::Modifier::modify(Raw* o) {
  return o;
}

birch::Type* birch::Modifier::modify(EmptyType* o) {
  return o;
}

birch::Type* birch::Modifier::modify(TypeList* o) {
  o->head = o->head->accept(this);
  o->tail = o->tail->accept(this);
  return o;
}

birch::Type* birch::Modifier::modify(NamedType* o) {
  o->typeArgs = o->typeArgs->accept(this);
  return o;
}

birch::Type* birch::Modifier::modify(MemberType* o) {
  o->left = o->left->accept(this);
  o->right = o->right->accept(this);
  return o;
}

birch::Type* birch::Modifier::modify(ArrayType* o) {
  o->single = o->single->accept(this);
  return o;
}

birch::Type* birch::Modifier::modify(TupleType* o) {
  o->single = o->single->accept(this);
  return o;
}

birch::Type* birch::Modifier::modify(FunctionType* o) {
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  return o;
}

birch::Type* birch::Modifier::modify(OptionalType* o) {
  o->single = o->single->accept(this);
  return o;
}
