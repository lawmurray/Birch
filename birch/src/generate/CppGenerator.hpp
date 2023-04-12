/**
 * @file
 */
#pragma once

#include "src/common/TypeParameterised.hpp"
#include "src/generate/IndentableGenerator.hpp"

namespace birch {
/**
 * C++ code generator.
 *
 * @ingroup io
 */
class CppGenerator: public IndentableGenerator {
public:
  /**
   * Constructor.
   *
   * @param base Base stream.
   * @param level Indentation level.
   * @param header Output header instead of source?
   * @param includeInline Include inline classes and functions?
   * @param includeLines Include #line annotations?
   */
  CppGenerator(std::ostream& base, const int level, const bool header,
      const bool includeInline, const bool includeLines);

  using IndentableGenerator::visit;

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
  virtual void visit(const LocalVariable* o);
  virtual void visit(const TupleVariable* o);
  virtual void visit(const Function* o);
  virtual void visit(const Program* o);
  virtual void visit(const BinaryOperator* o);
  virtual void visit(const UnaryOperator* o);
  virtual void visit(const Basic* o);
  virtual void visit(const Class* o);
  virtual void visit(const Struct* o);
  virtual void visit(const Generic* o);
  virtual void visit(const Braces* o);
  virtual void visit(const ExpressionStatement* o);
  virtual void visit(const If* o);
  virtual void visit(const For* o);
  virtual void visit(const Parallel* o);
  virtual void visit(const While* o);
  virtual void visit(const DoWhile* o);
  virtual void visit(const With* o);
  virtual void visit(const Assert* o);
  virtual void visit(const Return* o);
  virtual void visit(const Factor* o);
  virtual void visit(const Raw* o);
  virtual void visit(const StatementList* o);

  virtual void visit(const EmptyType* o);
  virtual void visit(const ArrayType* o);
  virtual void visit(const TupleType* o);
  virtual void visit(const OptionalType* o);
  virtual void visit(const FutureType* o);
  virtual void visit(const MemberType* o);
  virtual void visit(const NamedType* o);
  virtual void visit(const TypeList* o);
  virtual void visit(const DeducedType* o);

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
   * Generate a documentation comment.
   */
  void genDoc(const Location* loc);

  /**
   * Generate macro to update line source line only. This does not generate
   * any executable code, and so is suitable for use in e.g. initializer
   * lists.
   */
  void genSourceLine(const Location* loc);

  /**
   * Generate the name of a loop index.
   */
  std::string genIndex(const Statement* o);

  /**
   * Output header instead of source?
   */
  bool header;

  /**
   * Include inline classes and functions?
   */
  bool includeInline;

  /**
   * Include #line annotations?
   */
  bool includeLines;

  /**
   * Are we on the left side of an assignment statement?
   */
  int inAssign;

  /**
   * Are we in the declaration of a global variable?
   */
  int inGlobal;

  /**
   * Are we inside a constructor?
   */
  int inConstructor;

  /**
   * Are we inside the body of an operator?
   */
  int inOperator;

  /**
   * Are we inside the body of a lambda function?
   */
  int inLambda;

  /**
   * Are we on the right side of a member expression?
   */
  int inMember;

  /**
   * Are we in a sequence?
   */
  int inSequence;

  /**
   * Are we in a return statement?
   */
  int inReturn;
};
}

template<class T>
void birch::CppGenerator::genInit(const T* o) {
  if (!o->brackets->isEmpty()) {
    if (!o->value->isEmpty()) {
      middle("(numbirch::make_shape(" << o->brackets << "), " << o->value << ')');
    } else if (!o->args->isEmpty()) {
      middle("(numbirch::make_shape(" << o->brackets << "), " << o->args << ')');
    } else {
      middle("(numbirch::make_shape(" << o->brackets << "))");
    }
  } else if (!o->value->isEmpty()) {
    if (!inConstructor) {
      middle(" = ");
    }
    if (*o->op == "~") {
      middle('(' << o->value << ")->random()");
    } else if (*o->op == "<~") {
      middle('(' << o->value << ")->variate()");
    } else {
      middle(o->value);
    }
  } else if (!o->args->isEmpty()) {
    if (!inConstructor) {
      middle('(');
    }
    middle("std::in_place, " << o->args);
    if (!inConstructor) {
      middle(')');
    }
  }
}

template<class ObjectType>
void birch::CppGenerator::genTemplateParams(const ObjectType* o) {
  if (!o->typeParams->isEmpty()) {
    genSourceLine(o->loc);
    start("template<");
    for (auto iter = o->typeParams->begin(); iter != o->typeParams->end();
        ++iter) {
      if (iter != o->typeParams->begin()) {
        middle(", ");
      }
      middle("class " << *iter);
    }
    finish('>');
  }
}

template<class ObjectType>
void birch::CppGenerator::genTemplateArgs(const ObjectType* o) {
  if (!o->typeParams->isEmpty()) {
    middle('<' << o->typeParams << '>');
  }
}
