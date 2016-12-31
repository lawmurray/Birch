/**
 * @file
 */
#include "bi/visitor/Visitor.hpp"

void bi::Visitor::visit(const EmptyExpression* o) {
  //
}

void bi::Visitor::visit(const EmptyStatement* o) {
  //
}

void bi::Visitor::visit(const EmptyType* o) {
  //
}

void bi::Visitor::visit(const BoolLiteral* o) {
  o->type->accept(this);
}

void bi::Visitor::visit(const IntLiteral* o) {
  o->type->accept(this);
}

void bi::Visitor::visit(const RealLiteral* o) {
  o->type->accept(this);
}

void bi::Visitor::visit(const StringLiteral* o) {
  o->type->accept(this);
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

void bi::Visitor::visit(const ExpressionList* o) {
  o->head->accept(this);
  o->tail->accept(this);
}

void bi::Visitor::visit(const StatementList* o) {
  o->head->accept(this);
  o->tail->accept(this);
}

void bi::Visitor::visit(const ParenthesesExpression* o) {
  o->expr->accept(this);
}

void bi::Visitor::visit(const BracesExpression* o) {
  o->stmt->accept(this);
}

void bi::Visitor::visit(const BracketsExpression* o) {
  o->expr->accept(this);
  o->brackets->accept(this);
}

void bi::Visitor::visit(const Range* o) {
  o->left->accept(this);
  o->right->accept(this);
}

void bi::Visitor::visit(const This* o) {
  //
}

void bi::Visitor::visit(const Traversal* o) {
  o->left->accept(this);
  o->right->accept(this);
}

void bi::Visitor::visit(const VarReference* o) {
  o->name->accept(this);
}

void bi::Visitor::visit(const FuncReference* o) {
  o->name->accept(this);
  o->parens->accept(this);
}

void bi::Visitor::visit(const RandomReference* o) {
  o->name->accept(this);
}

void bi::Visitor::visit(const ModelReference* o) {
  o->name->accept(this);
  o->brackets->accept(this);
}

void bi::Visitor::visit(const ProgReference* o) {
  o->name->accept(this);
  o->parens->accept(this);
}

void bi::Visitor::visit(const VarParameter* o) {
  o->name->accept(this);
  o->type->accept(this);
  o->value->accept(this);
}

void bi::Visitor::visit(const FuncParameter* o) {
  o->name->accept(this);
  o->parens->accept(this);
  o->result->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const RandomParameter* o) {
  o->left->accept(this);
  o->right->accept(this);
}

void bi::Visitor::visit(const ModelParameter* o) {
  o->name->accept(this);
  o->parens->accept(this);
  o->base->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const ProgParameter* o) {
  o->name->accept(this);
  o->parens->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const File* o) {
  o->imports->accept(this);
  o->root->accept(this);
}

void bi::Visitor::visit(const Import* o) {
  o->path->accept(this);
}

void bi::Visitor::visit(const ExpressionStatement* o) {
  o->expr->accept(this);
}

void bi::Visitor::visit(const Conditional* o) {
  o->cond->accept(this);
  o->braces->accept(this);
  o->falseBraces->accept(this);
}

void bi::Visitor::visit(const Loop* o) {
  o->cond->accept(this);
  o->braces->accept(this);
}

void bi::Visitor::visit(const Raw* o) {
  //
}

void bi::Visitor::visit(const VarDeclaration* o) {
  o->param->accept(this);
}

void bi::Visitor::visit(const FuncDeclaration* o) {
  o->param->accept(this);
}

void bi::Visitor::visit(const ModelDeclaration* o) {
  o->param->accept(this);
}

void bi::Visitor::visit(const ProgDeclaration* o) {
  o->param->accept(this);
}

void bi::Visitor::visit(const ParenthesesType* o) {
  o->type->accept(this);
}

void bi::Visitor::visit(const RandomType* o) {
  o->left->accept(this);
  o->right->accept(this);
}

void bi::Visitor::visit(const TypeList* o) {
  o->head->accept(this);
  o->tail->accept(this);
}
