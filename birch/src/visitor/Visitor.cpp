/**
 * @file
 */
#include "src/visitor/Visitor.hpp"

void birch::Visitor::visit(const Package* o) {
  for (auto file : o->sources) {
    file->accept(this);
  }
}

void birch::Visitor::visit(const File* o) {
  o->root->accept(this);
}

void birch::Visitor::visit(const Name* o) {
  //
}

void birch::Visitor::visit(const EmptyExpression* o) {
  //
}

void birch::Visitor::visit(const ExpressionList* o) {
  o->head->accept(this);
  o->tail->accept(this);
}

void birch::Visitor::visit(const Literal<bool>* o) {
  //
}

void birch::Visitor::visit(const Literal<int64_t>* o) {
  //
}

void birch::Visitor::visit(const Literal<double>* o) {
  //
}

void birch::Visitor::visit(const Literal<const char*>* o) {
  //
}

void birch::Visitor::visit(const Parentheses* o) {
  o->single->accept(this);
}

void birch::Visitor::visit(const Sequence* o) {
  o->single->accept(this);
}

void birch::Visitor::visit(const Cast* o) {
  o->returnType->accept(this);
  o->single->accept(this);
}

void birch::Visitor::visit(const Call* o) {
  o->single->accept(this);
  o->args->accept(this);
}

void birch::Visitor::visit(const BinaryCall* o) {
  o->left->accept(this);
  o->right->accept(this);
}

void birch::Visitor::visit(const UnaryCall* o) {
  o->single->accept(this);
}

void birch::Visitor::visit(const Assign* o) {
  o->left->accept(this);
  o->right->accept(this);
}

void birch::Visitor::visit(const Slice* o) {
  o->single->accept(this);
  o->brackets->accept(this);
}

void birch::Visitor::visit(const Query* o) {
  o->single->accept(this);
}

void birch::Visitor::visit(const Get* o) {
  o->single->accept(this);
}

void birch::Visitor::visit(const LambdaFunction* o) {
  o->params->accept(this);
  o->returnType->accept(this);
  o->braces->accept(this);
}

void birch::Visitor::visit(const Span* o) {
  o->single->accept(this);
}

void birch::Visitor::visit(const Range* o) {
  o->left->accept(this);
  o->right->accept(this);
}

void birch::Visitor::visit(const Global* o) {
  o->single->accept(this);
}

void birch::Visitor::visit(const Super* o) {
  //
}

void birch::Visitor::visit(const This* o) {
  //
}

void birch::Visitor::visit(const Nil* o) {
  //
}

void birch::Visitor::visit(const Member* o) {
  o->left->accept(this);
  o->right->accept(this);
}

void birch::Visitor::visit(const Parameter* o) {
  o->type->accept(this);
  o->value->accept(this);
}

void birch::Visitor::visit(const NamedExpression* o) {
  o->typeArgs->accept(this);
}

void birch::Visitor::visit(const EmptyStatement* o) {
  //
}

void birch::Visitor::visit(const Braces* o) {
  o->single->accept(this);
}

void birch::Visitor::visit(const StatementList* o) {
  o->head->accept(this);
  o->tail->accept(this);
}

void birch::Visitor::visit(const GlobalVariable* o) {
  o->type->accept(this);
  o->brackets->accept(this);
  o->args->accept(this);
  o->value->accept(this);
}

void birch::Visitor::visit(const MemberVariable* o) {
  o->type->accept(this);
  o->brackets->accept(this);
  o->args->accept(this);
  o->value->accept(this);
}

void birch::Visitor::visit(const MemberPhantom* o) {
  //
}

void birch::Visitor::visit(const LocalVariable* o) {
  o->type->accept(this);
  o->brackets->accept(this);
  o->args->accept(this);
  o->value->accept(this);
}

void birch::Visitor::visit(const TupleVariable* o) {
  o->locals->accept(this);
  o->value->accept(this);
}

void birch::Visitor::visit(const Function* o) {
  o->typeParams->accept(this);
  o->params->accept(this);
  o->returnType->accept(this);
  o->braces->accept(this);
}

void birch::Visitor::visit(const Program* o) {
  o->params->accept(this);
  o->braces->accept(this);
}

void birch::Visitor::visit(const MemberFunction* o) {
  o->typeParams->accept(this);
  o->params->accept(this);
  o->returnType->accept(this);
  o->braces->accept(this);
}

void birch::Visitor::visit(const BinaryOperator* o) {
  o->typeParams->accept(this);
  o->left->accept(this);
  o->right->accept(this);
  o->returnType->accept(this);
  o->braces->accept(this);
}

void birch::Visitor::visit(const UnaryOperator* o) {
  o->typeParams->accept(this);
  o->single->accept(this);
  o->returnType->accept(this);
  o->braces->accept(this);
}

void birch::Visitor::visit(const AssignmentOperator* o) {
  o->single->accept(this);
  o->braces->accept(this);
}

void birch::Visitor::visit(const ConversionOperator* o) {
  o->returnType->accept(this);
  o->braces->accept(this);
}

void birch::Visitor::visit(const SliceOperator* o) {
  o->params->accept(this);
  o->returnType->accept(this);
  o->braces->accept(this);
}

void birch::Visitor::visit(const Class* o) {
  o->typeParams->accept(this);
  o->params->accept(this);
  o->base->accept(this);
  o->args->accept(this);
  o->braces->accept(this);
}

void birch::Visitor::visit(const Struct* o) {
  o->typeParams->accept(this);
  o->params->accept(this);
  o->base->accept(this);
  o->args->accept(this);
  o->braces->accept(this);
}

void birch::Visitor::visit(const Basic* o) {
  o->typeParams->accept(this);
  o->base->accept(this);
}

void birch::Visitor::visit(const Generic* o) {
  //
}

void birch::Visitor::visit(const ExpressionStatement* o) {
  o->single->accept(this);
}

void birch::Visitor::visit(const If* o) {
  o->cond->accept(this);
  o->braces->accept(this);
  o->falseBraces->accept(this);
}

void birch::Visitor::visit(const For* o) {
  o->index->accept(this);
  o->from->accept(this);
  o->to->accept(this);
  o->braces->accept(this);
}

void birch::Visitor::visit(const Parallel* o) {
  o->index->accept(this);
  o->from->accept(this);
  o->to->accept(this);
  o->braces->accept(this);
}

void birch::Visitor::visit(const While* o) {
  o->cond->accept(this);
  o->braces->accept(this);
}

void birch::Visitor::visit(const DoWhile* o) {
  o->braces->accept(this);
  o->cond->accept(this);
}

void birch::Visitor::visit(const With* o) {
  o->single->accept(this);
  o->braces->accept(this);
}

void birch::Visitor::visit(const Block* o) {
  o->braces->accept(this);
}

void birch::Visitor::visit(const Assert* o) {
  o->cond->accept(this);
}

void birch::Visitor::visit(const Return* o) {
  o->single->accept(this);
}

void birch::Visitor::visit(const Factor* o) {
  o->single->accept(this);
}

void birch::Visitor::visit(const Raw* o) {
  //
}

void birch::Visitor::visit(const EmptyType* o) {
  //
}

void birch::Visitor::visit(const TypeList* o) {
  o->head->accept(this);
  o->tail->accept(this);
}

void birch::Visitor::visit(const NamedType* o) {
  o->typeArgs->accept(this);
}

void birch::Visitor::visit(const MemberType* o) {
  o->left->accept(this);
  o->right->accept(this);
}

void birch::Visitor::visit(const ArrayType* o) {
  o->single->accept(this);
}

void birch::Visitor::visit(const TupleType* o) {
  o->single->accept(this);
}

void birch::Visitor::visit(const OptionalType* o) {
  o->single->accept(this);
}

void birch::Visitor::visit(const FutureType* o) {
  o->single->accept(this);
}

void birch::Visitor::visit(const DeducedType* o) {
  //
}
