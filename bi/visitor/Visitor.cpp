/**
 * @file
 */
#include "bi/visitor/Visitor.hpp"

bi::Visitor::~Visitor() {
  //
}

void bi::Visitor::visit(const File* o) {
  o->root->accept(this);
}

void bi::Visitor::visit(const Name* o) {
  //
}

void bi::Visitor::visit(const Path* o) {
  o->head->accept(this);
  if (o->tail) {
    o->tail->accept(this);
  }
}

void bi::Visitor::visit(const EmptyExpression* o) {
  //
}

void bi::Visitor::visit(const List<Expression>* o) {
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

void bi::Visitor::visit(const ParenthesesExpression* o) {
  o->single->accept(this);
}

void bi::Visitor::visit(const BracketsExpression* o) {
  o->single->accept(this);
  o->brackets->accept(this);
}

void bi::Visitor::visit(const LambdaFunction* o) {
  o->parens->accept(this);
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

void bi::Visitor::visit(const Member* o) {
  o->left->accept(this);
  o->right->accept(this);
}

void bi::Visitor::visit(const Parameter* o) {
  o->type->accept(this);
  o->value->accept(this);
}

void bi::Visitor::visit(const Identifier<Unknown>* o) {
  o->parens->accept(this);
}

void bi::Visitor::visit(const Identifier<Parameter>* o) {
  o->parens->accept(this);
}

void bi::Visitor::visit(const Identifier<GlobalVariable>* o) {
  o->parens->accept(this);
}

void bi::Visitor::visit(const Identifier<LocalVariable>* o) {
  o->parens->accept(this);
}

void bi::Visitor::visit(const Identifier<MemberVariable>* o) {
  o->parens->accept(this);
}

void bi::Visitor::visit(const Identifier<Function>* o) {
  o->parens->accept(this);
}

void bi::Visitor::visit(const Identifier<Coroutine>* o) {
  o->parens->accept(this);
}

void bi::Visitor::visit(const Identifier<MemberFunction>* o) {
  o->parens->accept(this);
}

void bi::Visitor::visit(const Identifier<BinaryOperator>* o) {
  o->left->accept(this);
  o->right->accept(this);
}

void bi::Visitor::visit(const Identifier<UnaryOperator>* o) {
  o->single->accept(this);
}


void bi::Visitor::visit(const EmptyStatement* o) {
  //
}

void bi::Visitor::visit(const List<Statement>* o) {
  o->head->accept(this);
  o->tail->accept(this);
}

void bi::Visitor::visit(const Assignment* o) {
  o->left->accept(this);
  o->right->accept(this);
}

void bi::Visitor::visit(const GlobalVariable* o) {
  o->type->accept(this);
}

void bi::Visitor::visit(const LocalVariable* o) {
  o->type->accept(this);
}

void bi::Visitor::visit(const MemberVariable* o) {
  o->type->accept(this);
}

void bi::Visitor::visit(const Function* o) {
  o->parens->accept(this);
  o->returnType->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const Coroutine* o) {
  o->parens->accept(this);
  o->returnType->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const Program* o) {
  o->parens->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const MemberFunction* o) {
  o->parens->accept(this);
  o->returnType->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const BinaryOperator* o) {
  o->left->accept(this);
  o->right->accept(this);
  o->returnType->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const UnaryOperator* o) {
  o->single->accept(this);
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
  o->base->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const AliasType* o) {
  o->base->accept(this);
}

void bi::Visitor::visit(const BasicType* o) {
  //
}

void bi::Visitor::visit(const Import* o) {
  o->path->accept(this);
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

void bi::Visitor::visit(const Return* o) {
  o->single->accept(this);
}

void bi::Visitor::visit(const Raw* o) {
  //
}

void bi::Visitor::visit(const EmptyType* o) {
  //
}

void bi::Visitor::visit(const List<Type>* o) {
  o->head->accept(this);
  o->tail->accept(this);
}
void bi::Visitor::visit(const IdentifierType<Class>* o) {
  //
}

void bi::Visitor::visit(const IdentifierType<AliasType>* o) {
  //
}

void bi::Visitor::visit(const IdentifierType<BasicType>* o) {
  //
}

void bi::Visitor::visit(const BracketsType* o) {
  o->single->accept(this);
  o->brackets->accept(this);
}

void bi::Visitor::visit(const ParenthesesType* o) {
  o->single->accept(this);
}

void bi::Visitor::visit(const FunctionType* o) {
  o->parens->accept(this);
  o->type->accept(this);
}

void bi::Visitor::visit(const CoroutineType* o) {
  o->type->accept(this);
}

