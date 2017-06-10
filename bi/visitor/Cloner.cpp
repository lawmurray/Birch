/**
 * @file
 */
#include "bi/visitor/Cloner.hpp"

bi::Cloner::~Cloner() {
  //
}

bi::Expression* bi::Cloner::clone(const EmptyExpression* o) {
  return new EmptyExpression();
}

bi::Statement* bi::Cloner::clone(const EmptyStatement* o) {
  return new EmptyStatement();
}

bi::Type* bi::Cloner::clone(const EmptyType* o) {
  return new EmptyType(o->assignable);
}

bi::Expression* bi::Cloner::clone(const BooleanLiteral* o) {
  return new BooleanLiteral(o->value, o->str, o->type->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const IntegerLiteral* o) {
  return new IntegerLiteral(o->value, o->str, o->type->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const RealLiteral* o) {
  return new RealLiteral(o->value, o->str, o->type->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const StringLiteral* o) {
  return new StringLiteral(o->value, o->str, o->type->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const ExpressionList* o) {
  return new ExpressionList(o->head->accept(this), o->tail->accept(this),
      o->loc);
}

bi::Statement* bi::Cloner::clone(const StatementList* o) {
  return new StatementList(o->head->accept(this), o->tail->accept(this),
      o->loc);
}

bi::Expression* bi::Cloner::clone(const ParenthesesExpression* o) {
  return new ParenthesesExpression(o->single->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const BracesExpression* o) {
  return new BracesExpression(o->single->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const BracketsExpression* o) {
  return new BracketsExpression(o->single->accept(this),
      o->brackets->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Span* o) {
  return new Span(o->single->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Index* o) {
  return new Index(o->single->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Range* o) {
  return new Range(o->left->accept(this), o->right->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Member* o) {
  return new Member(o->left->accept(this), o->right->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const This* o) {
  return new This(o->loc);
}

bi::Expression* bi::Cloner::clone(const Super* o) {
  return new Super(o->loc);
}

bi::Expression* bi::Cloner::clone(const VarReference* o) {
  return new VarReference(o->name, o->loc);
}

bi::Expression* bi::Cloner::clone(const FuncReference* o) {
  return new FuncReference(o->name, o->parens->accept(this), o->loc);
}

bi::Type* bi::Cloner::clone(const TypeReference* o) {
  return new TypeReference(o->name, o->loc, o->assignable);
}

bi::Prog* bi::Cloner::clone(const ProgReference* o) {
  return new ProgReference(o->name, o->parens->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const VarParameter* o) {
  return new VarParameter(o->name, o->type->accept(this), o->form,
      o->parens->accept(this), o->value->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const FuncParameter* o) {
  return new FuncParameter(o->name, o->parens->accept(this),
      o->type->accept(this), o->braces->accept(this), o->form, o->loc);
}

bi::Expression* bi::Cloner::clone(const ConversionParameter* o) {
  return new ConversionParameter(o->type->accept(this),
      o->braces->accept(this), o->loc);
}

bi::Type* bi::Cloner::clone(const TypeParameter* o) {
  return new TypeParameter(o->name, o->parens->accept(this),
      o->base->accept(this), o->baseParens->accept(this),
      o->braces->accept(this), o->form, o->loc);
}

bi::Prog* bi::Cloner::clone(const ProgParameter* o) {
  return new ProgParameter(o->name, o->parens->accept(this),
      o->braces->accept(this), o->loc);
}

bi::File* bi::Cloner::clone(const File* o) {
  return new File(o->path, o->root->accept(this));
}

bi::Statement* bi::Cloner::clone(const Import* o) {
  return new Import(o->path, o->file, o->loc);
}

bi::Statement* bi::Cloner::clone(const ExpressionStatement* o) {
  return new ExpressionStatement(o->single->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const If* o) {
  return new If(o->cond->accept(this), o->braces->accept(this),
      o->falseBraces->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const For* o) {
  return new For(o->index->accept(this), o->from->accept(this),
      o->to->accept(this), o->braces->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const While* o) {
  return new While(o->cond->accept(this), o->braces->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const Return* o) {
  return new Return(o->single->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const Raw* o) {
  return new Raw(o->name, o->raw, o->loc);
}

bi::Statement* bi::Cloner::clone(const VarDeclaration* o) {
  return new VarDeclaration(
      dynamic_cast<VarParameter*>(o->param->accept(this)), o->loc);
}

bi::Statement* bi::Cloner::clone(const FuncDeclaration* o) {
  return new FuncDeclaration(
      dynamic_cast<FuncParameter*>(o->param->accept(this)), o->loc);
}

bi::Statement* bi::Cloner::clone(const ConversionDeclaration* o) {
  return new ConversionDeclaration(
      dynamic_cast<ConversionParameter*>(o->param->accept(this)), o->loc);
}

bi::Statement* bi::Cloner::clone(const TypeDeclaration* o) {
  return new TypeDeclaration(
      dynamic_cast<TypeParameter*>(o->param->accept(this)), o->loc);
}

bi::Statement* bi::Cloner::clone(const ProgDeclaration* o) {
  return new ProgDeclaration(
      dynamic_cast<ProgParameter*>(o->param->accept(this)), o->loc);
}

bi::Type* bi::Cloner::clone(const BracketsType* o) {
  return new BracketsType(o->single->accept(this), o->ndims, o->loc,
      o->assignable);
}

bi::Type* bi::Cloner::clone(const ParenthesesType* o) {
  return new ParenthesesType(o->single->accept(this), o->loc, o->assignable);
}

bi::Type* bi::Cloner::clone(const FunctionType* o) {
  return new FunctionType(o->parens->accept(this), o->type->accept(this),
      o->loc, o->assignable);
}

bi::Type* bi::Cloner::clone(const CoroutineType* o) {
  return new CoroutineType(o->type->accept(this), o->loc, o->assignable);
}

bi::Type* bi::Cloner::clone(const TypeList* o) {
  TypeList* result = new TypeList(o->head->accept(this),
      o->tail->accept(this), o->loc);
  result->assignable = o->assignable;
  return result;
}
