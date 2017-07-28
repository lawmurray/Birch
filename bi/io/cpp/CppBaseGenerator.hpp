/**
 * @file
 */
#pragma once

#include "bi/io/indentable_ostream.hpp"

namespace bi {
/**
 * C++ code generator for within-function code.
 *
 * @ingroup compiler_io
 */
class CppBaseGenerator: public indentable_ostream {
public:
  CppBaseGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

  using indentable_ostream::visit;

  virtual void visit(const Name* o);

  virtual void visit(const List<Expression>* o);
  virtual void visit(const Literal<bool>* o);
  virtual void visit(const Literal<int64_t>* o);
  virtual void visit(const Literal<double>* o);
  virtual void visit(const Literal<const char*>* o);
  virtual void visit(const Parentheses* o);
  virtual void visit(const Brackets* o);
  virtual void visit(const Call* o);
  virtual void visit(const BinaryCall* o);
  virtual void visit(const UnaryCall* o);
  virtual void visit(const Slice* o);
  virtual void visit(const LambdaFunction* o);
  virtual void visit(const Span* o);
  virtual void visit(const Index* o);
  virtual void visit(const Range* o);
  virtual void visit(const Member* o);
  virtual void visit(const Super* o);
  virtual void visit(const This* o);
  virtual void visit(const Parameter* o);
  virtual void visit(const MemberParameter* o);
  virtual void visit(const Identifier<Parameter>* o);
  virtual void visit(const Identifier<MemberParameter>* o);
  virtual void visit(const Identifier<GlobalVariable>* o);
  virtual void visit(const Identifier<LocalVariable>* o);
  virtual void visit(const Identifier<MemberVariable>* o);
  virtual void visit(const OverloadedIdentifier<Function>* o);
  virtual void visit(const OverloadedIdentifier<Coroutine>* o);
  virtual void visit(const OverloadedIdentifier<MemberFunction>* o);
  virtual void visit(const OverloadedIdentifier<MemberCoroutine>* o);

  virtual void visit(const File* o);
  virtual void visit(const Import* o);
  virtual void visit(const GlobalVariable* o);
  virtual void visit(const LocalVariable* o);
  virtual void visit(const MemberVariable* o);
  virtual void visit(const List<Statement>* o);
  virtual void visit(const Function* o);
  virtual void visit(const Coroutine* o);
  virtual void visit(const MemberFunction* o);
  virtual void visit(const MemberCoroutine* o);
  virtual void visit(const Program* o);
  virtual void visit(const BinaryOperator* o);
  virtual void visit(const UnaryOperator* o);
  virtual void visit(const AssignmentOperator* o);
  virtual void visit(const ConversionOperator* o);
  virtual void visit(const Basic* o);
  virtual void visit(const Class* o);
  virtual void visit(const Alias* o);
  virtual void visit(const Assignment* o);
  virtual void visit(const ExpressionStatement* o);
  virtual void visit(const If* o);
  virtual void visit(const For* o);
  virtual void visit(const While* o);
  virtual void visit(const Assert* o);
  virtual void visit(const Return* o);
  virtual void visit(const Yield* o);
  virtual void visit(const Raw* o);

  virtual void visit(const List<Type>* o);
  virtual void visit(const EmptyType* o);
  virtual void visit(const ArrayType* o);
  virtual void visit(const ParenthesesType* o);
  virtual void visit(const FunctionType* o);
  virtual void visit(const FiberType* o);
  virtual void visit(const ClassType* o);
  virtual void visit(const BasicType* o);
  virtual void visit(const AliasType* o);

protected:
  /**
   * Generate arguments to a function.
   *
   * @param args Arguments.
   * @param params Parameters
   */
  void genArgs(const Expression* args, const Expression* params);

  /**
   * Generate an argument to a function.
   *
   * @param arg The argument.
   * @param param The parameter.
   */
  void genArg(const Expression* arg, const Expression* param);

  /**
   * Generate the initialization of a variable, including the call to the
   * constructor and/or assignment of the initial value.
   */
  template<class T>
  void genInit(const T* o);

  /**
   * Output header instead of source?
   */
  bool header;
};
}

template<class T>
void bi::CppBaseGenerator::genInit(const T* o) {
  if (o->type->isArray()) {
    ArrayType* type = dynamic_cast<ArrayType*>(o->type->strip());
    assert(type);
    middle("(make_frame(" << type->brackets << ')');
    if (!o->parens->isEmpty()) {
      middle(", " << o->parens->strip());
    }
    middle(')');
  } else if (o->type->isClass()) {
    ClassType* type = dynamic_cast<ClassType*>(o->type->strip());
    assert(type);
    if (!o->value->isEmpty()) {
      middle('(');
    }
    middle(" = make_object<bi::type::" << type->name << '>');
    if (o->parens->isEmpty()) {
      middle("()");
    } else {
      ///@todo Use genArgs()
      middle(o->parens);
    }
    if (!o->value->isEmpty()) {
      middle(')');
    }
  }
  if (!o->value->isEmpty()) {
    middle(" = " << o->value);
  }
}
