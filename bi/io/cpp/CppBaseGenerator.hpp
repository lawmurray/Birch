/**
 * @file
 */
#pragma once

#include "bi/common/TypeParameterised.hpp"
#include "bi/io/indentable_ostream.hpp"

namespace bi {
/**
 * C++ code generator.
 *
 * @ingroup io
 */
class CppBaseGenerator: public indentable_ostream {
public:
  /**
   * Constructor.
   *
   * @param base Base stream.
   * @param level Indentation level.
   * @param header Output header instead of course?
   * @param generic Include generic classes, functions and fibers?
   */
  CppBaseGenerator(std::ostream& base, const int level = 0,
      const bool header = false, const bool generic = false);

  using indentable_ostream::visit;

  virtual void visit(const Name* o);

  virtual void visit(const ExpressionList* o);
  virtual void visit(const Literal<bool>* o);
  virtual void visit(const Literal<int64_t>* o);
  virtual void visit(const Literal<double>* o);
  virtual void visit(const Literal<const char*>* o);
  virtual void visit(const Parentheses* o);
  virtual void visit(const Sequence* o);
  virtual void visit(const Cast* o);
  virtual void visit(const Call* o);
  virtual void visit(const BinaryCall* o);
  virtual void visit(const UnaryCall* o);
  virtual void visit(const Assign* o);
  virtual void visit(const Slice* o);
  virtual void visit(const Query* o);
  virtual void visit(const Get* o);
  virtual void visit(const LambdaFunction* o);
  virtual void visit(const Span* o);
  virtual void visit(const Index* o);
  virtual void visit(const Range* o);
  virtual void visit(const Member* o);
  virtual void visit(const Global* o);
  virtual void visit(const This* o);
  virtual void visit(const Super* o);
  virtual void visit(const Nil* o);
  virtual void visit(const Parameter* o);
  virtual void visit(const NamedExpression* o);

  virtual void visit(const File* o);
  virtual void visit(const GlobalVariable* o);
  virtual void visit(const MemberVariable* o);
  virtual void visit(const LocalVariable* o);
  virtual void visit(const Function* o);
  virtual void visit(const Fiber* o);
  virtual void visit(const MemberFunction* o);
  virtual void visit(const MemberFiber* o);
  virtual void visit(const Program* o);
  virtual void visit(const BinaryOperator* o);
  virtual void visit(const UnaryOperator* o);
  virtual void visit(const AssignmentOperator* o);
  virtual void visit(const ConversionOperator* o);
  virtual void visit(const Basic* o);
  virtual void visit(const Class* o);
  virtual void visit(const Generic* o);
  virtual void visit(const Assume* o);
  virtual void visit(const ExpressionStatement* o);
  virtual void visit(const If* o);
  virtual void visit(const For* o);
  virtual void visit(const Parallel* o);
  virtual void visit(const While* o);
  virtual void visit(const DoWhile* o);
  virtual void visit(const Assert* o);
  virtual void visit(const Return* o);
  virtual void visit(const Yield* o);
  virtual void visit(const Raw* o);
  virtual void visit(const StatementList* o);

  virtual void visit(const EmptyType* o);
  virtual void visit(const ArrayType* o);
  virtual void visit(const TupleType* o);
  virtual void visit(const FunctionType* o);
  virtual void visit(const FiberType* o);
  virtual void visit(const OptionalType* o);
  virtual void visit(const MemberType* o);
  virtual void visit(const NamedType* o);
  virtual void visit(const TypeList* o);

protected:
  /**
   * Generate code for template parameters (`template<...>`).
   */
  template<class ObjectType>
  void genTemplateParams(const ObjectType* o);

  /**
   * Generate code for template arguments (`<...>`).
   */
  template<class ObjectType>
  void genTemplateArgs(const ObjectType* o);

  /**
   * Generate the initialization of a variable, including the call to the
   * constructor and/or assignment of the initial value.
   */
  template<class T>
  void genInit(const T* o);

  /**
   * Generate macro to put function call on stack trace.
   */
  void genTraceFunction(const std::string& name, const Location* loc);

  /**
   * Generate macro to update line on stack trace.
   */
  void genTraceLine(const Location* loc);

  /**
   * Generate macro to update line source line only. This does not generate
   * any executable code, and so is suitable for use in e.g. initializer
   * lists.
   */
  void genSourceLine(const Location* loc);

  /**
   * Output header instead of source?
   */
  bool header;

  /**
   * Include generic classes, functions and fibers?
   */
  bool generic;

  /**
   * Are we on the left side of an assignment statement?
   */
  int inAssign;

  /**
   * Are we inside a constructor?
   */
  int inConstructor;

  /**
   * Are we inside the body of a lambda function?
   */
  int inLambda;

  /**
   * Are we on the right side of a member expression?
   */
  int inMember;
};
}

template<class T>
void bi::CppBaseGenerator::genInit(const T* o) {
  if (!o->brackets->isEmpty()) {
    middle('(');
    middle("libbirch::make_shape(" << o->brackets << ')');
    if (!o->args->isEmpty()) {
      middle(", " << o->args);
    } else if (!o->value->isEmpty()) {
      middle(", " << o->value);
    }
    middle(')');
  } else if (!o->args->isEmpty()) {
    middle('(' << o->args << ')');
  } else if (!o->value->isEmpty()) {
    middle(" = " << o->value);
  }
}

template<class ObjectType>
void bi::CppBaseGenerator::genTemplateParams(const ObjectType* o) {
  if (!o->typeParams->isEmpty()) {
    if (!header) {
      genSourceLine(o->loc);
    }
    start("template<");
    for (auto iter = o->typeParams->begin(); iter != o->typeParams->end();
        ++iter) {
      if (iter != o->typeParams->begin()) {
        middle(',');
      }
      middle("class " << *iter);
    }
    finish('>');
  }
}

template<class ObjectType>
void bi::CppBaseGenerator::genTemplateArgs(const ObjectType* o) {
  if (!o->typeParams->isEmpty()) {
    middle('<' << o->typeParams << '>');
  }
}
