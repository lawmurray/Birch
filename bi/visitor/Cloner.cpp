/**
 * @file
 */
#include "bi/visitor/Cloner.hpp"

bi::Cloner::~Cloner() {
  //
}

bi::Package* bi::Cloner::clone(const Package* o) {
  return new Package(o->name, o->headers, o->sources);
}

bi::File* bi::Cloner::clone(const File* o) {
  return new File(o->path, o->scope, o->root->accept(this));
}

bi::Expression* bi::Cloner::clone(const EmptyExpression* o) {
  return new EmptyExpression(o->loc);
}

bi::Expression* bi::Cloner::clone(const ExpressionList* o) {
  return new ExpressionList(o->head->accept(this), o->tail->accept(this),
      o->loc);
}

bi::Expression* bi::Cloner::clone(const Literal<bool>* o) {
  return new Literal<bool>(o->value, o->str, o->type->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Literal<int64_t>* o) {
  return new Literal<int64_t>(o->value, o->str, o->type->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Literal<double>* o) {
  return new Literal<double>(o->value, o->str, o->type->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Literal<const char*>* o) {
  return new Literal<const char*>(o->value, o->str, o->type->accept(this),
      o->loc);
}

bi::Expression* bi::Cloner::clone(const Parentheses* o) {
  return new Parentheses(o->single->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Sequence* o) {
  return new Sequence(o->single->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Binary* o) {
  return new Binary(o->left->accept(this), o->right->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Cast* o) {
  return new Cast(o->returnType->accept(this), o->single->accept(this),
      o->loc);
}

bi::Expression* bi::Cloner::clone(const Call* o) {
  return new Call(o->single->accept(this), o->args->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const BinaryCall* o) {
  return new BinaryCall(o->single->accept(this), o->args->accept(this),
      o->loc);
}

bi::Expression* bi::Cloner::clone(const UnaryCall* o) {
  return new UnaryCall(o->single->accept(this), o->args->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Slice* o) {
  return new Slice(o->single->accept(this), o->brackets->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Query* o) {
  return new Query(o->single->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Get* o) {
  return new Get(o->single->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const LambdaFunction* o) {
  return new LambdaFunction(o->params->accept(this),
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

bi::Expression* bi::Cloner::clone(const Nil* o) {
  return new Nil(o->loc);
}

bi::Expression* bi::Cloner::clone(const LocalVariable* o) {
  return new LocalVariable(o->name, o->type->accept(this),
      o->brackets->accept(this), o->args->accept(this),
      o->value->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Parameter* o) {
  return new Parameter(o->name, o->type->accept(this), o->value->accept(this),
      o->loc);
}

bi::Expression* bi::Cloner::clone(const Generic* o) {
  return new Generic(o->name, o->type->accept(this), o->loc);
}

bi::Expression* bi::Cloner::clone(const Identifier<Unknown>* o) {
  return new Identifier<Unknown>(o->name, o->loc);
}

bi::Expression* bi::Cloner::clone(const Identifier<Parameter>* o) {
  return new Identifier<Parameter>(o->name, o->loc);
}

bi::Expression* bi::Cloner::clone(const Identifier<GlobalVariable>* o) {
  return new Identifier<GlobalVariable>(o->name, o->loc);
}

bi::Expression* bi::Cloner::clone(const Identifier<LocalVariable>* o) {
  return new Identifier<LocalVariable>(o->name, o->loc);
}

bi::Expression* bi::Cloner::clone(const Identifier<MemberVariable>* o) {
  return new Identifier<MemberVariable>(o->name, o->loc);
}

bi::Expression* bi::Cloner::clone(const OverloadedIdentifier<Function>* o) {
  return new OverloadedIdentifier<Function>(o->name, o->loc);
}

bi::Expression* bi::Cloner::clone(const OverloadedIdentifier<Fiber>* o) {
  return new OverloadedIdentifier<Fiber>(o->name, o->loc);
}

bi::Expression* bi::Cloner::clone(
    const OverloadedIdentifier<MemberFunction>* o) {
  return new OverloadedIdentifier<MemberFunction>(o->name, o->loc);
}

bi::Expression* bi::Cloner::clone(
    const OverloadedIdentifier<MemberFiber>* o) {
  return new OverloadedIdentifier<MemberFiber>(o->name, o->loc);
}

bi::Expression* bi::Cloner::clone(
    const OverloadedIdentifier<BinaryOperator>* o) {
  return new OverloadedIdentifier<BinaryOperator>(o->name, o->loc);
}

bi::Expression* bi::Cloner::clone(
    const OverloadedIdentifier<UnaryOperator>* o) {
  return new OverloadedIdentifier<UnaryOperator>(o->name, o->loc);
}

bi::Statement* bi::Cloner::clone(const EmptyStatement* o) {
  return new EmptyStatement();
}

bi::Statement* bi::Cloner::clone(const Braces* o) {
  return new Braces(o->single->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const StatementList* o) {
  return new StatementList(o->head->accept(this), o->tail->accept(this),
      o->loc);
}

bi::Statement* bi::Cloner::clone(const Assignment* o) {
  return new Assignment(o->left->accept(this), o->name,
      o->right->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const GlobalVariable* o) {
  return new GlobalVariable(o->name, o->type->accept(this),
      o->brackets->accept(this), o->args->accept(this),
      o->value->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const MemberVariable* o) {
  return new MemberVariable(o->annotation, o->name, o->type->accept(this),
      o->brackets->accept(this), o->args->accept(this),
      o->value->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const Function* o) {
  return new Function(o->annotation, o->name, o->params->accept(this),
      o->returnType->accept(this), o->braces->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const Fiber* o) {
  return new Fiber(o->annotation, o->name, o->params->accept(this),
      o->returnType->accept(this), o->braces->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const Program* o) {
  return new Program(o->name, o->params->accept(this),
      o->braces->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const MemberFunction* o) {
  return new MemberFunction(o->annotation, o->name, o->params->accept(this),
      o->returnType->accept(this), o->braces->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const MemberFiber* o) {
  return new MemberFiber(o->annotation, o->name, o->params->accept(this),
      o->returnType->accept(this), o->braces->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const BinaryOperator* o) {
  return new BinaryOperator(o->name, o->params->accept(this),
      o->returnType->accept(this), o->braces->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const UnaryOperator* o) {
  return new UnaryOperator(o->name, o->params->accept(this),
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

bi::Statement* bi::Cloner::clone(const Class* o) {
  return new Class(o->name, o->typeParams->accept(this),
      o->params->accept(this), o->base->accept(this), o->alias,
      o->args->accept(this), o->braces->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const Basic* o) {
  return new Basic(o->name, o->base->accept(this), o->alias, o->loc);
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

bi::Statement* bi::Cloner::clone(const DoWhile* o) {
  return new DoWhile(o->braces->accept(this), o->cond->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const Assert* o) {
  return new Assert(o->cond->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const Return* o) {
  return new Return(o->single->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const Yield* o) {
  return new Yield(o->single->accept(this), o->loc);
}

bi::Statement* bi::Cloner::clone(const Raw* o) {
  return new Raw(o->name, o->raw, o->loc);
}

bi::Type* bi::Cloner::clone(const AnyType* o) {
  return new AnyType(o->loc);
}

bi::Type* bi::Cloner::clone(const EmptyType* o) {
  return new EmptyType(o->loc);
}

bi::Type* bi::Cloner::clone(const TypeList* o) {
  return new TypeList(o->head->accept(this), o->tail->accept(this), o->loc);
}

bi::Type* bi::Cloner::clone(const UnknownType* o) {
  return new UnknownType(o->weak, o->name, o->typeArgs->accept(this), o->read,
      o->loc);
}

bi::Type* bi::Cloner::clone(const ClassType* o) {
  return new ClassType(o->name, o->typeArgs->accept(this), o->loc);
}

bi::Type* bi::Cloner::clone(const BasicType* o) {
  return new BasicType(o->name, o->loc);
}

bi::Type* bi::Cloner::clone(const GenericType* o) {
  return new GenericType(o->name, o->loc);
}

bi::Type* bi::Cloner::clone(const ArrayType* o) {
  return new ArrayType(o->single->accept(this), o->ndims, o->loc);
}

bi::Type* bi::Cloner::clone(const TupleType* o) {
  return new TupleType(o->single->accept(this), o->loc);
}

bi::Type* bi::Cloner::clone(const SequenceType* o) {
  return new SequenceType(o->single->accept(this), o->loc);
}

bi::Type* bi::Cloner::clone(const BinaryType* o) {
  return new BinaryType(o->left->accept(this), o->right->accept(this), o->loc);
}

bi::Type* bi::Cloner::clone(const FunctionType* o) {
  return new FunctionType(o->params->accept(this),
      o->returnType->accept(this), o->loc);
}

bi::Type* bi::Cloner::clone(const OverloadedType* o) {
  return new OverloadedType(o->overloaded, o->loc);
}

bi::Type* bi::Cloner::clone(const FiberType* o) {
  return new FiberType(o->single->accept(this), o->loc);
}

bi::Type* bi::Cloner::clone(const OptionalType* o) {
  return new OptionalType(o->single->accept(this), o->loc);
}

bi::Type* bi::Cloner::clone(const PointerType* o) {
  return new PointerType(o->weak, o->single->accept(this), o->read, o->loc);
}

bi::Type* bi::Cloner::clone(const NilType* o) {
  return new NilType(o->loc);
}
