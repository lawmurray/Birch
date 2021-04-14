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
   */
  CppGenerator(std::ostream& base, const int level = 0,
      const bool header = false, const bool includeInline = false);

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
  virtual void visit(const MemberVariable* o);
  virtual void visit(const LocalVariable* o);
  virtual void visit(const TupleVariable* o);
  virtual void visit(const Function* o);
  virtual void visit(const MemberFunction* o);
  virtual void visit(const Program* o);
  virtual void visit(const BinaryOperator* o);
  virtual void visit(const UnaryOperator* o);
  virtual void visit(const Basic* o);
  virtual void visit(const Class* o);
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
  virtual void visit(const MemberType* o);
  virtual void visit(const NamedType* o);
  virtual void visit(const TypeList* o);
  virtual void visit(const TypeOf* o);

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
   * Generate the name of a loop index.
   */
  virtual std::string getIndex(const Statement* o);

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
   * Include inline classes and functions?
   */
  bool includeInline;

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
      middle("libbirch::make_array_from_value<" << o->type->element() << ">(");
      middle("libbirch::make_shape(" << o->brackets << ')');
      middle(", " << o->value << ')');
    } else if (!o->args->isEmpty()) {
      middle("libbirch::make_array<" << o->type->element() << ">(");
      middle("libbirch::make_shape(" << o->brackets << ')');
      middle(", std::in_place," << o->args << ')');
    } else {
      middle("libbirch::make_array<" << o->type->element() << ">(");
      middle("libbirch::make_shape(" << o->brackets << "))");
    }
  } else if (!o->value->isEmpty()) {
    if (*o->op == "<~") {
      middle("birch::handle_simulate(" << o->value << ')');
    } else if (*o->op == "~") {
      middle("birch::handle_assume(" << o->value << ')');
    } else {
      middle(o->value);
    }
  } else if (!o->args->isEmpty()) {
    middle("libbirch::make<" << o->type << ">(std::in_place, " << o->args << ')');
  } else {
    middle("libbirch::make<" << o->type << ">()");
  }
}

template<class ObjectType>
void birch::CppGenerator::genTemplateParams(const ObjectType* o) {
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
void birch::CppGenerator::genTemplateArgs(const ObjectType* o) {
  if (!o->typeParams->isEmpty()) {
    middle('<' << o->typeParams << '>');
  }
}
