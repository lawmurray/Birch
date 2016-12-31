/**
 * @file
 */
#include "bi/visitor/Modifier.hpp"

bi::Expression* bi::Modifier::modify(EmptyExpression* o) {
  return o;
}

bi::Statement* bi::Modifier::modify(EmptyStatement* o) {
  return o;
}

bi::Type* bi::Modifier::modify(EmptyType* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(BoolLiteral* o) {
  o->type = o->type->acceptModify(this);
  return o;
}

bi::Expression* bi::Modifier::modify(IntLiteral* o) {
  o->type = o->type->acceptModify(this);
  return o;
}

bi::Expression* bi::Modifier::modify(RealLiteral* o) {
  o->type = o->type->acceptModify(this);
  return o;
}

bi::Expression* bi::Modifier::modify(StringLiteral* o) {
  o->type = o->type->acceptModify(this);
  return o;
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

bi::Expression* bi::Modifier::modify(ExpressionList* o) {
  o->head = o->head->acceptModify(this);
  o->tail = o->tail->acceptModify(this);
  return o;
}

bi::Statement* bi::Modifier::modify(StatementList* o) {
  o->head = o->head->acceptModify(this);
  o->tail = o->tail->acceptModify(this);
  return o;
}

bi::Expression* bi::Modifier::modify(ParenthesesExpression* o) {
  o->expr = o->expr->acceptModify(this);
  return o;
}

bi::Expression* bi::Modifier::modify(BracesExpression* o) {
  o->stmt = o->stmt->acceptModify(this);
  return o;
}

bi::Expression* bi::Modifier::modify(BracketsExpression* o) {
  o->expr = o->expr->acceptModify(this);
  o->brackets = o->brackets->acceptModify(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Range* o) {
  o->left = o->left->acceptModify(this);
  o->right = o->right->acceptModify(this);
  return o;
}

bi::Expression* bi::Modifier::modify(Traversal* o) {
  o->left = o->left->acceptModify(this);
  o->right = o->right->acceptModify(this);
  return o;
}

bi::Expression* bi::Modifier::modify(This* o) {
  return o;
}

bi::Expression* bi::Modifier::modify(VarReference* o) {
  o->name->acceptModify(this);
  return o;
}

bi::Expression* bi::Modifier::modify(FuncReference* o) {
  o->name->acceptModify(this);
  o->parens = o->parens->acceptModify(this);
  return o;
}

bi::Expression* bi::Modifier::modify(RandomReference* o) {
  o->name->acceptModify(this);
  return o;
}

bi::Type* bi::Modifier::modify(ModelReference* o) {
  o->name->acceptModify(this);
  o->brackets = o->brackets->acceptModify(this);
  return o;
}

bi::Prog* bi::Modifier::modify(ProgReference* o) {
  o->name->acceptModify(this);
  o->parens = o->parens->acceptModify(this);
  return o;
}

bi::Expression* bi::Modifier::modify(VarParameter* o) {
  o->name->acceptModify(this);
  o->type = o->type->acceptModify(this);
  o->value = o->value->acceptModify(this);
  return o;
}

bi::Expression* bi::Modifier::modify(FuncParameter* o) {
  o->name->acceptModify(this);
  o->parens = o->parens->acceptModify(this);
  o->result = o->result->acceptModify(this);
  o->braces = o->braces->acceptModify(this);
  return o;
}

bi::Expression* bi::Modifier::modify(RandomParameter* o) {
  o->left = o->left->acceptModify(this);
  o->op->acceptModify(this);
  o->right = o->right->acceptModify(this);
  o->pull = o->pull->acceptModify(this);
  o->push = o->push->acceptModify(this);
  return o;
}

bi::Type* bi::Modifier::modify(ModelParameter* o) {
  o->name->acceptModify(this);
  o->parens = o->parens->acceptModify(this);
  o->base = o->base->acceptModify(this);
  o->braces = o->braces->acceptModify(this);
  return o;
}

bi::Prog* bi::Modifier::modify(ProgParameter* o) {
  o->name->acceptModify(this);
  o->parens = o->parens->acceptModify(this);
  o->braces = o->braces->acceptModify(this);
  return o;
}

void bi::Modifier::modify(File* o) {
  o->imports = o->imports->acceptModify(this);
  o->root = o->root->acceptModify(this);
}

bi::Statement* bi::Modifier::modify(Import* o) {
  o->path->acceptModify(this);
  return o;
}

bi::Statement* bi::Modifier::modify(ExpressionStatement* o) {
  o->expr = o->expr->acceptModify(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Conditional* o) {
  o->cond = o->cond->acceptModify(this);
  o->braces = o->braces->acceptModify(this);
  o->falseBraces = o->falseBraces->acceptModify(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Loop* o) {
  o->cond = o->cond->acceptModify(this);
  o->braces = o->braces->acceptModify(this);
  return o;
}

bi::Statement* bi::Modifier::modify(Raw* o) {
  return o;
}

bi::Statement* bi::Modifier::modify(VarDeclaration* o) {
  o->param = dynamic_cast<VarParameter*>(o->param->acceptModify(this));
  assert(o->param);
  return o;
}

bi::Statement* bi::Modifier::modify(FuncDeclaration* o) {
  o->param = dynamic_cast<FuncParameter*>(o->param->acceptModify(this));
  assert(o->param);
  return o;
}

bi::Statement* bi::Modifier::modify(ModelDeclaration* o) {
  o->param = dynamic_cast<ModelParameter*>(o->param->acceptModify(this));
  assert(o->param);
  return o;
}

bi::Statement* bi::Modifier::modify(ProgDeclaration* o) {
  o->param = dynamic_cast<ProgParameter*>(o->param->acceptModify(this));
  assert(o->param);
  return o;
}

bi::Type* bi::Modifier::modify(ParenthesesType* o) {
  o->type = o->type->acceptModify(this);
  return o;
}

bi::Type* bi::Modifier::modify(RandomType* o) {
  o->left = o->left->acceptModify(this);
  o->right = o->right->acceptModify(this);
  return o;
}

bi::Type* bi::Modifier::modify(TypeList* o) {
  o->head = o->head->acceptModify(this);
  o->tail = o->tail->acceptModify(this);
  return o;
}
