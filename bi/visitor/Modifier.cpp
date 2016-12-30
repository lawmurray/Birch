/**
 * @file
 */
#include "bi/visitor/Modifier.hpp"

void bi::Modifier::modify(EmptyExpression* o) {
  //
}

void bi::Modifier::modify(EmptyStatement* o) {
  //
}

void bi::Modifier::modify(EmptyType* o) {
  //
}

void bi::Modifier::modify(BoolLiteral* o) {
  o->type->acceptModify(this);
}

void bi::Modifier::modify(IntLiteral* o) {
  o->type->acceptModify(this);
}

void bi::Modifier::modify(RealLiteral* o) {
  o->type->acceptModify(this);
}

void bi::Modifier::modify(StringLiteral* o) {
  o->type->acceptModify(this);
}

void bi::Modifier::modify(Name* o) {
  //
}

void bi::Modifier::modify(Path* o) {
  o->head->acceptModify(this);
  if (o->tail) {
    o->tail->acceptModify(this);
  }
}

void bi::Modifier::modify(ExpressionList* o) {
  o->head->acceptModify(this);
  o->tail->acceptModify(this);
}

void bi::Modifier::modify(StatementList* o) {
  o->head->acceptModify(this);
  o->tail->acceptModify(this);
}

void bi::Modifier::modify(ParenthesesExpression* o) {
  o->expr->acceptModify(this);
}

void bi::Modifier::modify(BracesExpression* o) {
  o->stmt->acceptModify(this);
}

void bi::Modifier::modify(BracketsExpression* o) {
  o->expr->acceptModify(this);
  o->brackets->acceptModify(this);
}

void bi::Modifier::modify(RandomVariable* o) {
  o->left->acceptModify(this);
  o->right->acceptModify(this);
}

void bi::Modifier::modify(Range* o) {
  o->left->acceptModify(this);
  o->right->acceptModify(this);
}

void bi::Modifier::modify(Traversal* o) {
  o->left->acceptModify(this);
  o->right->acceptModify(this);
}

void bi::Modifier::modify(This* o) {
  //
}

void bi::Modifier::modify(VarReference* o) {
  o->name->acceptModify(this);
}

void bi::Modifier::modify(FuncReference* o) {
  o->name->acceptModify(this);
  o->parens->acceptModify(this);
}

void bi::Modifier::modify(ModelReference* o) {
  o->name->acceptModify(this);
  o->brackets->acceptModify(this);
}

void bi::Modifier::modify(ProgReference* o) {
  o->name->acceptModify(this);
  o->parens->acceptModify(this);
}

void bi::Modifier::modify(VarParameter* o) {
  o->name->acceptModify(this);
  o->type->acceptModify(this);
  o->value->acceptModify(this);
}

void bi::Modifier::modify(FuncParameter* o) {
  o->name->acceptModify(this);
  o->parens->acceptModify(this);
  o->result->acceptModify(this);
  o->braces->acceptModify(this);
}

void bi::Modifier::modify(ModelParameter* o) {
  o->name->acceptModify(this);
  o->parens->acceptModify(this);
  o->base->acceptModify(this);
  o->braces->acceptModify(this);
}

void bi::Modifier::modify(ProgParameter* o) {
  o->name->acceptModify(this);
  o->parens->acceptModify(this);
  o->braces->acceptModify(this);
}

void bi::Modifier::modify(File* o) {
  o->imports->acceptModify(this);
  o->root->acceptModify(this);
}

void bi::Modifier::modify(Import* o) {
  o->path->acceptModify(this);
}

void bi::Modifier::modify(ExpressionStatement* o) {
  o->expr->acceptModify(this);
}

void bi::Modifier::modify(Conditional* o) {
  o->cond->acceptModify(this);
  o->braces->acceptModify(this);
  o->falseBraces->acceptModify(this);
}

void bi::Modifier::modify(Loop* o) {
  o->cond->acceptModify(this);
  o->braces->acceptModify(this);
}

void bi::Modifier::modify(Raw* o) {
  //
}

void bi::Modifier::modify(VarDeclaration* o) {
  o->param->acceptModify(this);
}

void bi::Modifier::modify(FuncDeclaration* o) {
  o->param->acceptModify(this);
}

void bi::Modifier::modify(ModelDeclaration* o) {
  o->param->acceptModify(this);
}

void bi::Modifier::modify(ProgDeclaration* o) {
  o->param->acceptModify(this);
}

void bi::Modifier::modify(ParenthesesType* o) {
  o->type->acceptModify(this);
}

void bi::Modifier::modify(RandomVariableType* o) {
  o->left->acceptModify(this);
  o->right->acceptModify(this);
}

void bi::Modifier::modify(TypeList* o) {
  o->head->acceptModify(this);
  o->tail->acceptModify(this);
}
