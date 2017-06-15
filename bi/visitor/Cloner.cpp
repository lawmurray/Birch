/**
 * @file
 */
#include "bi/visitor/Cloner.hpp"

bi::Cloner::~Cloner() {
  //
}

bi::File* bi::Cloner::clone(const File* o) {
  return new File(o->path, o->root->accept(this));
}

bi::Expression* bi::Cloner::clone(const EmptyExpression* o) {
  return new EmptyExpression();
}

bi::Expression* bi::Cloner::clone(const List<Expression>* o) {
  return new List<Expression>(o->head->accept(this), o->tail->accept(this),
      o->loc);
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

bi::Expression* bi::Cloner::clone(const LambdaFunction* o) {
  return new LambdaFunction(o->parens->accept(this),
      o->returnType->accept(this), o->braces->accept(this), o->loc);
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

bi::Expression* bi::Cloner::clone(const Parameter* o) {
  return new Parameter(o->name, o->type->accept(this), o->value->accept(this),
      o->loc);
}

bi::Expression* bi::Cloner::clone(const GlobalVariable* o) {
  return new GlobalVariable(o->name, o->type->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const LocalVariable* o) {
  return new GlobalVariable(o->name, o->type->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const MemberVariable* o) {
  return new GlobalVariable(o->name, o->type->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Identifier<Unknown>* o) {
  return new Identifier<Unknown>(o->name, o->parens->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Identifier<Parameter>* o) {
  return new Identifier<Parameter>(o->name, o->parens->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Identifier<GlobalVariable>* o) {
  return new Identifier<GlobalVariable>(o->name, o->parens->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Identifier<LocalVariable>* o) {
  return new Identifier<LocalVariable>(o->name, o->parens->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Identifier<MemberVariable>* o) {
  return new Identifier<MemberVariable>(o->name, o->parens->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Identifier<Function>* o) {
  return new Identifier<Function>(o->name, o->parens->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Identifier<Coroutine>* o) {
  return new Identifier<Coroutine>(o->name, o->parens->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Identifier<MemberFunction>* o) {
  return new Identifier<MemberFunction>(o->name, o->parens->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Identifier<BinaryOperator>* o) {
  return new Identifier<BinaryOperator>(o->left->accept(this), o->name,
      o->right->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Identifier<UnaryOperator>* o) {
  return new Identifier<UnaryOperator>(o->name, o->single->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const EmptyStatement* o) {
  return new EmptyStatement();
}

bi::Statement* bi::Cloner::clone(const List<Statement>* o) {
  return new List<Statement>(o->head->accept(this), o->tail->accept(this),
      o->loc);
}

bi::Statement* bi::Cloner::clone(const Assignment* o) {
  return new Assignment(o->left->accept(this), o->name,
      o->right->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const Function* o) {
  return new Function(o->name, o->parens->accept(this),
      o->returnType->accept(this), o->braces->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const Coroutine* o) {
  return new Coroutine(o->name, o->parens->accept(this),
      o->returnType->accept(this), o->braces->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const Program* o) {
  return new Program(o->name, o->parens->accept(this),
      o->braces->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const MemberFunction* o) {
  return new MemberFunction(o->name, o->parens->accept(this),
      o->returnType->accept(this), o->braces->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const BinaryOperator* o) {
  return new BinaryOperator(o->left->accept(this), o->name,
      o->right->accept(this), o->returnType->accept(this),
      o->braces->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const UnaryOperator* o) {
  return new UnaryOperator(o->name, o->single->accept(this),
      o->returnType->accept(this), o->braces->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const AssignmentOperator* o) {
  return new AssignmentOperator(o->name, o->single->accept(this),
      o->braces->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const ConversionOperator* o) {
  return new ConversionOperator(o->returnType->accept(this),
      o->braces->accept(this), o->loc);
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

bi::Type* bi::Cloner::clone(const EmptyType* o) {
  return new EmptyType(o->assignable);
}

bi::Type* bi::Cloner::clone(const List<Type>* o) {
  List<Type>* result = new List<Type>(o->head->accept(this),
      o->tail->accept(this), o->loc);
  result->assignable = o->assignable;
  return result;
}
bi::Type* bi::Cloner::clone(const TypeReference* o) {
  return new TypeReference(o->name, o->loc, o->assignable);
}

bi::Type* bi::Cloner::clone(const TypeParameter* o) {
  return new TypeParameter(o->name, o->parens->accept(this),
      o->base->accept(this), o->baseParens->accept(this),
      o->braces->accept(this), o->form, o->loc);
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

