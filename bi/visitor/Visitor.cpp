/**
 * @file
 */
#include "bi/visitor/Visitor.hpp"

bi::Visitor::~Visitor() {
  //
}

void bi::Visitor::visit(const Package* o) {
  for (auto file : o->files) {
    file->accept(this);
  }
}

void bi::Visitor::visit(const File* o) {
  o->root->accept(this);
}

void bi::Visitor::visit(const Name* o) {
  //
}

void bi::Visitor::visit(const EmptyExpression* o) {
  //
}

void bi::Visitor::visit(const ExpressionList* o) {
  o->head->accept(this);
  o->tail->accept(this);
}

void bi::Visitor::visit(const Literal<bool>* o) {
  o->type->accept(this);
}

void bi::Visitor::visit(const Literal<int64_t>* o) {
  o->type->accept(this);
}

void bi::Visitor::visit(const Literal<double>* o) {
  o->type->accept(this);
}

void bi::Visitor::visit(const Literal<const char*>* o) {
  o->type->accept(this);
}

void bi::Visitor::visit(const Parentheses* o) {
  o->single->accept(this);
}

void bi::Visitor::visit(const Sequence* o) {
  o->single->accept(this);
}

void bi::Visitor::visit(const Binary* o) {
  o->left->accept(this);
  o->right->accept(this);
}

void bi::Visitor::visit(const Cast* o) {
  o->returnType->accept(this);
  o->single->accept(this);
}

void bi::Visitor::visit(const Call* o) {
  o->single->accept(this);
  o->args->accept(this);
}

void bi::Visitor::visit(const BinaryCall* o) {
  o->single->accept(this);
  o->args->accept(this);
}

void bi::Visitor::visit(const UnaryCall* o) {
  o->single->accept(this);
  o->args->accept(this);
}

void bi::Visitor::visit(const Slice* o) {
  o->single->accept(this);
  o->brackets->accept(this);
}

void bi::Visitor::visit(const Query* o) {
  o->single->accept(this);
}

void bi::Visitor::visit(const Get* o) {
  o->single->accept(this);
}

void bi::Visitor::visit(const LambdaFunction* o) {
  o->params->accept(this);
  o->returnType->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const Span* o) {
  o->single->accept(this);
}

void bi::Visitor::visit(const Index* o) {
  o->single->accept(this);
}

void bi::Visitor::visit(const Range* o) {
  o->left->accept(this);
  o->right->accept(this);
}

void bi::Visitor::visit(const Super* o) {
  //
}

void bi::Visitor::visit(const This* o) {
  //
}

void bi::Visitor::visit(const Nil* o) {
  //
}

void bi::Visitor::visit(const Member* o) {
  o->left->accept(this);
  o->right->accept(this);
}

void bi::Visitor::visit(const LocalVariable* o) {
  o->type->accept(this);
  o->brackets->accept(this);
  o->args->accept(this);
  o->value->accept(this);
}

void bi::Visitor::visit(const Parameter* o) {
  o->type->accept(this);
  o->value->accept(this);
}


void bi::Visitor::visit(const Identifier<Unknown>* o) {
  //
}

void bi::Visitor::visit(const Identifier<Parameter>* o) {
  //
}

void bi::Visitor::visit(const Identifier<GlobalVariable>* o) {
  //
}

void bi::Visitor::visit(const Identifier<LocalVariable>* o) {
  //
}

void bi::Visitor::visit(const Identifier<MemberVariable>* o) {
  //
}

void bi::Visitor::visit(const OverloadedIdentifier<Function>* o) {
  //
}

void bi::Visitor::visit(const OverloadedIdentifier<Fiber>* o) {
  //
}

void bi::Visitor::visit(const OverloadedIdentifier<MemberFunction>* o) {
  //
}

void bi::Visitor::visit(const OverloadedIdentifier<MemberFiber>* o) {
  //
}

void bi::Visitor::visit(const OverloadedIdentifier<BinaryOperator>* o) {
  //
}

void bi::Visitor::visit(const OverloadedIdentifier<UnaryOperator>* o) {
  //
}

void bi::Visitor::visit(const EmptyStatement* o) {
  //
}

void bi::Visitor::visit(const Braces* o) {
  o->single->accept(this);
}

void bi::Visitor::visit(const StatementList* o) {
  o->head->accept(this);
  o->tail->accept(this);
}

void bi::Visitor::visit(const Assignment* o) {
  o->left->accept(this);
  o->right->accept(this);
}

void bi::Visitor::visit(const GlobalVariable* o) {
  o->type->accept(this);
  o->brackets->accept(this);
  o->args->accept(this);
  o->value->accept(this);
}

void bi::Visitor::visit(const MemberVariable* o) {
  o->type->accept(this);
  o->brackets->accept(this);
  o->args->accept(this);
  o->value->accept(this);
}

void bi::Visitor::visit(const Function* o) {
  o->params->accept(this);
  o->returnType->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const Fiber* o) {
  o->params->accept(this);
  o->returnType->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const Program* o) {
  o->params->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const MemberFunction* o) {
  o->params->accept(this);
  o->returnType->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const MemberFiber* o) {
  o->params->accept(this);
  o->returnType->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const BinaryOperator* o) {
  o->params->accept(this);
  o->returnType->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const UnaryOperator* o) {
  o->params->accept(this);
  o->returnType->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const AssignmentOperator* o) {
  o->single->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const ConversionOperator* o) {
  o->returnType->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const Class* o) {
  o->typeParams->accept(this);
  o->params->accept(this);
  o->base->accept(this);
  o->args->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const Basic* o) {
  o->base->accept(this);
}

void bi::Visitor::visit(const Generic* o) {
  //
}

void bi::Visitor::visit(const ExpressionStatement* o) {
  o->single->accept(this);
}

void bi::Visitor::visit(const If* o) {
  o->cond->accept(this);
  o->braces->accept(this);
  o->falseBraces->accept(this);
}

void bi::Visitor::visit(const For* o) {
  o->index->accept(this);
  o->from->accept(this);
  o->to->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const While* o) {
  o->cond->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const DoWhile* o) {
  o->braces->accept(this);
  o->cond->accept(this);
}

void bi::Visitor::visit(const Assert* o) {
  o->cond->accept(this);
}

void bi::Visitor::visit(const Return* o) {
  o->single->accept(this);
}

void bi::Visitor::visit(const Yield* o) {
  o->single->accept(this);
}

void bi::Visitor::visit(const Raw* o) {
  //
}

void bi::Visitor::visit(const AnyType* o) {
  //
}

void bi::Visitor::visit(const EmptyType* o) {
  //
}

void bi::Visitor::visit(const TypeList* o) {
  o->head->accept(this);
  o->tail->accept(this);
}

void bi::Visitor::visit(const UnknownType* o) {
  o->typeArgs->accept(this);
}

void bi::Visitor::visit(const ClassType* o) {
  o->typeArgs->accept(this);
}

void bi::Visitor::visit(const BasicType* o) {
  //
}

void bi::Visitor::visit(const GenericType* o) {
  //
}

void bi::Visitor::visit(const ArrayType* o) {
  o->single->accept(this);
}

void bi::Visitor::visit(const TupleType* o) {
  o->single->accept(this);
}

void bi::Visitor::visit(const SequenceType* o) {
  o->single->accept(this);
}

void bi::Visitor::visit(const BinaryType* o) {
  o->left->accept(this);
  o->right->accept(this);
}

void bi::Visitor::visit(const FunctionType* o) {
  o->params->accept(this);
  o->returnType->accept(this);
}

void bi::Visitor::visit(const OverloadedType* o) {
  //
}

void bi::Visitor::visit(const FiberType* o) {
  o->single->accept(this);
}

void bi::Visitor::visit(const OptionalType* o) {
  o->single->accept(this);
}

void bi::Visitor::visit(const PointerType* o) {
  o->single->accept(this);
}

void bi::Visitor::visit(const NilType* o) {
  //
}
