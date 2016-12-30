/**
 * @file
 */
#include "bi/visitor/Cloner.hpp"

bi::Expression* bi::Cloner::clone(const EmptyExpression* o) {
  return new EmptyExpression();
}

bi::Statement* bi::Cloner::clone(const EmptyStatement* o) {
  return new EmptyStatement();
}

bi::Type* bi::Cloner::clone(const EmptyType* o) {
  return new EmptyType();
}

bi::Expression* bi::Cloner::clone(const BoolLiteral* o) {
  return new BoolLiteral(o->value, o->str, o->type->acceptClone(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const IntLiteral* o) {
  return new IntLiteral(o->value, o->str, o->type->acceptClone(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const RealLiteral* o) {
  return new RealLiteral(o->value, o->str, o->type->acceptClone(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const StringLiteral* o) {
  return new StringLiteral(o->value, o->str, o->type->acceptClone(this),
      o->loc);
}

bi::Expression* bi::Cloner::clone(const ExpressionList* o) {
  return new ExpressionList(o->head->acceptClone(this),
      o->tail->acceptClone(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const StatementList* o) {
  return new StatementList(o->head->acceptClone(this),
      o->tail->acceptClone(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const ParenthesesExpression* o) {
  return new ParenthesesExpression(o->expr->acceptClone(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const BracesExpression* o) {
  return new BracesExpression(o->stmt->acceptClone(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const BracketsExpression* o) {
  return new BracketsExpression(o->expr->acceptClone(this),
      o->brackets->acceptClone(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const RandomVariable* o) {
  return new RandomVariable(o->left->acceptClone(this),
      o->right->acceptClone(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Range* o) {
  return new Range(o->left->acceptClone(this), o->right->acceptClone(this),
      o->loc);
}

bi::Expression* bi::Cloner::clone(const Traversal* o) {
  return new Traversal(o->left->acceptClone(this),
      o->right->acceptClone(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const This* o) {
  return new This(o->loc);
}

bi::Expression* bi::Cloner::clone(const VarReference* o) {
  return new VarReference(o->name, o->loc);
}

bi::Expression* bi::Cloner::clone(const FuncReference* o) {
  return new FuncReference(o->name, o->parens->acceptClone(this), o->form,
      o->loc);
}

bi::Type* bi::Cloner::clone(const ModelReference* o) {
  return new ModelReference(o->name, o->brackets->acceptClone(this), o->loc);
}

bi::Prog* bi::Cloner::clone(const ProgReference* o) {
  return new ProgReference(o->name, o->parens->acceptClone(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const VarParameter* o) {
  return new VarParameter(o->name, o->type->acceptClone(this),
      o->parens->acceptClone(this), o->value->acceptClone(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const FuncParameter* o) {
  return new FuncParameter(o->name, o->parens->acceptClone(this),
      o->result->acceptClone(this), o->braces->acceptClone(this), o->form,
      o->loc);
}

bi::Type* bi::Cloner::clone(const ModelParameter* o) {
  return new ModelParameter(o->name, o->parens->acceptClone(this), o->op,
      o->base->acceptClone(this), o->braces->acceptClone(this), o->loc);
}

bi::Prog* bi::Cloner::clone(const ProgParameter* o) {
  return new ProgParameter(o->name, o->parens->acceptClone(this),
      o->braces->acceptClone(this), o->loc);
}

bi::File* bi::Cloner::clone(const File* o) {
  return new File(o->path, o->imports->acceptClone(this),
      o->root->acceptClone(this));
}

bi::Statement* bi::Cloner::clone(const Import* o) {
  return new Import(o->path, o->file, o->loc);
}

bi::Statement* bi::Cloner::clone(const ExpressionStatement* o) {
  return new ExpressionStatement(o->expr->acceptClone(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const Conditional* o) {
  return new Conditional(o->cond->acceptClone(this),
      o->braces->acceptClone(this), o->falseBraces->acceptClone(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const Loop* o) {
  return new Loop(o->cond->acceptClone(this), o->braces->acceptClone(this),
      o->loc);
}

bi::Statement* bi::Cloner::clone(const Raw* o) {
  return new Raw(o->name, o->raw, o->loc);
}

bi::Statement* bi::Cloner::clone(const VarDeclaration* o) {
  return new VarDeclaration(
      dynamic_cast<VarParameter*>(o->param->acceptClone(this)), o->loc);
}

bi::Statement* bi::Cloner::clone(const FuncDeclaration* o) {
  return new FuncDeclaration(
      dynamic_cast<FuncParameter*>(o->param->acceptClone(this)), o->loc);
}

bi::Statement* bi::Cloner::clone(const ModelDeclaration* o) {
  return new ModelDeclaration(
      dynamic_cast<ModelParameter*>(o->param->acceptClone(this)), o->loc);
}

bi::Statement* bi::Cloner::clone(const ProgDeclaration* o) {
  return new ProgDeclaration(
      dynamic_cast<ProgParameter*>(o->param->acceptClone(this)), o->loc);
}

bi::Type* bi::Cloner::clone(const ParenthesesType* o) {
  return new ParenthesesType(o->type->acceptClone(this), o->loc);
}

bi::Type* bi::Cloner::clone(const RandomVariableType* o) {
  return new RandomVariableType(o->left->acceptClone(this),
      o->right->acceptClone(this), o->loc);
}

bi::Type* bi::Cloner::clone(const TypeList* o) {
  return new TypeList(o->head->acceptClone(this), o->tail->acceptClone(this),
      o->loc);
}
