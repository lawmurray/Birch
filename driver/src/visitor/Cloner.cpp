/**
 * @file
 */
#include "src/visitor/Cloner.hpp"

birch::Cloner::~Cloner() {
  //
}

birch::Package* birch::Cloner::clone(const Package* o) {
  return new Package(o->name, o->headers, o->sources);
}

birch::File* birch::Cloner::clone(const File* o) {
  return new File(o->path, o->root->accept(this));
}

birch::Expression* birch::Cloner::clone(const EmptyExpression* o) {
  return new EmptyExpression(o->loc);
}

birch::Expression* birch::Cloner::clone(const ExpressionList* o) {
  return new ExpressionList(o->head->accept(this), o->tail->accept(this),
      o->loc);
}

birch::Expression* birch::Cloner::clone(const Literal<bool>* o) {
  return new Literal<bool>(o->str, o->loc);
}

birch::Expression* birch::Cloner::clone(const Literal<int64_t>* o) {
  return new Literal<int64_t>(o->str, o->loc);
}

birch::Expression* birch::Cloner::clone(const Literal<double>* o) {
  return new Literal<double>(o->str, o->loc);
}

birch::Expression* birch::Cloner::clone(const Literal<const char*>* o) {
  return new Literal<const char*>(o->str, o->loc);
}

birch::Expression* birch::Cloner::clone(const Parentheses* o) {
  return new Parentheses(o->single->accept(this), o->loc);
}

birch::Expression* birch::Cloner::clone(const Sequence* o) {
  return new Sequence(o->single->accept(this), o->loc);
}

birch::Expression* birch::Cloner::clone(const Cast* o) {
  return new Cast(o->returnType->accept(this), o->single->accept(this),
      o->loc);
}

birch::Expression* birch::Cloner::clone(const Call* o) {
  return new Call(o->single->accept(this), o->args->accept(this), o->loc);
}

birch::Expression* birch::Cloner::clone(const BinaryCall* o) {
  return new BinaryCall(o->left->accept(this), o->name,
      o->right->accept(this), o->loc);
}

birch::Expression* birch::Cloner::clone(const UnaryCall* o) {
  return new UnaryCall(o->name, o->single->accept(this), o->loc);
}

birch::Expression* birch::Cloner::clone(const Assign* o) {
  return new Assign(o->left->accept(this), o->name, o->right->accept(this),
      o->loc);
}

birch::Expression* birch::Cloner::clone(const Slice* o) {
  return new Slice(o->single->accept(this), o->brackets->accept(this),
      o->loc);
}

birch::Expression* birch::Cloner::clone(const Query* o) {
  return new Query(o->single->accept(this), o->loc);
}

birch::Expression* birch::Cloner::clone(const Get* o) {
  return new Get(o->single->accept(this), o->loc);
}

birch::Expression* birch::Cloner::clone(const LambdaFunction* o) {
  return new LambdaFunction(o->params->accept(this),
      o->returnType->accept(this), o->braces->accept(this), o->loc);
}

birch::Expression* birch::Cloner::clone(const Span* o) {
  return new Span(o->single->accept(this), o->loc);
}

birch::Expression* birch::Cloner::clone(const Index* o) {
  return new Index(o->single->accept(this), o->loc);
}

birch::Expression* birch::Cloner::clone(const Range* o) {
  return new Range(o->left->accept(this), o->right->accept(this), o->loc);
}

birch::Expression* birch::Cloner::clone(const Member* o) {
  return new Member(o->left->accept(this), o->right->accept(this), o->loc);
}

birch::Expression* birch::Cloner::clone(const This* o) {
  return new This(o->loc);
}

birch::Expression* birch::Cloner::clone(const Super* o) {
  return new Super(o->loc);
}

birch::Expression* birch::Cloner::clone(const Global* o) {
  return new Global(o->single->accept(this), o->loc);
}

birch::Expression* birch::Cloner::clone(const Nil* o) {
  return new Nil(o->loc);
}

birch::Expression* birch::Cloner::clone(const Parameter* o) {
  return new Parameter(o->annotation, o->name, o->type->accept(this),
      o->value->accept(this), o->loc);
}

birch::Expression* birch::Cloner::clone(const Generic* o) {
  return new Generic(o->annotation, o->name, o->type->accept(this), o->loc);
}

birch::Expression* birch::Cloner::clone(const NamedExpression* o) {
  return new NamedExpression(o->name, o->typeArgs->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const EmptyStatement* o) {
  return new EmptyStatement();
}

birch::Statement* birch::Cloner::clone(const Braces* o) {
  return new Braces(o->single->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const StatementList* o) {
  return new StatementList(o->head->accept(this), o->tail->accept(this),
      o->loc);
}

birch::Statement* birch::Cloner::clone(const Assume* o) {
  return new Assume(o->left->accept(this), o->name, o->right->accept(this),
      o->loc);
}

birch::Statement* birch::Cloner::clone(const GlobalVariable* o) {
  return new GlobalVariable(o->annotation, o->name, o->type->accept(this),
      o->brackets->accept(this), o->args->accept(this),
      o->value->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const MemberVariable* o) {
  return new MemberVariable(o->annotation, o->name, o->type->accept(this),
      o->brackets->accept(this), o->args->accept(this),
      o->value->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const LocalVariable* o) {
  return new LocalVariable(o->annotation, o->name, o->type->accept(this),
      o->brackets->accept(this), o->args->accept(this),
      o->value->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const Function* o) {
  return new Function(o->annotation, o->name, o->typeParams->accept(this),
      o->params->accept(this), o->returnType->accept(this),
      o->braces->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const Program* o) {
  return new Program(o->name, o->params->accept(this),
      o->braces->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const MemberFunction* o) {
  return new MemberFunction(o->annotation, o->name,
      o->typeParams->accept(this), o->params->accept(this),
      o->returnType->accept(this), o->braces->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const BinaryOperator* o) {
  return new BinaryOperator(o->annotation, o->left->accept(this), o->name,
      o->right->accept(this), o->returnType->accept(this),
      o->braces->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const UnaryOperator* o) {
  return new UnaryOperator(o->annotation, o->name, o->single->accept(this),
      o->returnType->accept(this), o->braces->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const AssignmentOperator* o) {
  return new AssignmentOperator(o->single->accept(this),
      o->braces->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const ConversionOperator* o) {
  return new ConversionOperator(o->returnType->accept(this),
      o->braces->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const SliceOperator* o) {
  return new SliceOperator(o->params->accept(this),
      o->returnType->accept(this), o->braces->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const Class* o) {
  return new Class(o->annotation, o->name, o->typeParams->accept(this),
      o->params->accept(this), o->base->accept(this), o->alias,
      o->args->accept(this), o->braces->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const Basic* o) {
  return new Basic(o->annotation, o->name, o->typeParams->accept(this),
      o->base->accept(this), o->alias, o->loc);
}

birch::Statement* birch::Cloner::clone(const ExpressionStatement* o) {
  return new ExpressionStatement(o->single->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const If* o) {
  return new If(o->cond->accept(this), o->braces->accept(this),
      o->falseBraces->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const For* o) {
  return new For(o->annotation, o->index->accept(this),
      o->from->accept(this), o->to->accept(this), o->braces->accept(this),
      o->loc);
}

birch::Statement* birch::Cloner::clone(const Parallel* o) {
  return new Parallel(o->annotation, o->index->accept(this),
      o->from->accept(this), o->to->accept(this), o->braces->accept(this),
      o->loc);
}

birch::Statement* birch::Cloner::clone(const While* o) {
  return new While(o->cond->accept(this), o->braces->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const DoWhile* o) {
  return new DoWhile(o->braces->accept(this), o->cond->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const With* o) {
  return new With(o->single->accept(this), o->braces->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const Block* o) {
  return new Block(o->braces->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const Assert* o) {
  return new Assert(o->cond->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const Return* o) {
  return new Return(o->single->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const Factor* o) {
  return new Factor(o->single->accept(this), o->loc);
}

birch::Statement* birch::Cloner::clone(const Raw* o) {
  return new Raw(o->name, o->raw, o->loc);
}

birch::Type* birch::Cloner::clone(const EmptyType* o) {
  return new EmptyType(o->loc);
}

birch::Type* birch::Cloner::clone(const TypeList* o) {
  return new TypeList(o->head->accept(this), o->tail->accept(this), o->loc);
}

birch::Type* birch::Cloner::clone(const NamedType* o) {
  return new NamedType(o->name, o->typeArgs->accept(this), o->loc);
}

birch::Type* birch::Cloner::clone(const MemberType* o) {
  return new MemberType(o->left->accept(this), o->right->accept(this),
      o->loc);
}

birch::Type* birch::Cloner::clone(const ArrayType* o) {
  return new ArrayType(o->single->accept(this), o->ndims, o->loc);
}

birch::Type* birch::Cloner::clone(const TupleType* o) {
  return new TupleType(o->single->accept(this), o->loc);
}

birch::Type* birch::Cloner::clone(const FunctionType* o) {
  return new FunctionType(o->params->accept(this),
      o->returnType->accept(this), o->loc);
}

birch::Type* birch::Cloner::clone(const OptionalType* o) {
  return new OptionalType(o->single->accept(this), o->loc);
}
