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
  CppBaseGenerator(std::ostream& base, const int level = 0,
      const bool header = false);

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
  virtual void visit(const Identifier<Unknown>* o);
  virtual void visit(const Identifier<Parameter>* o);
  virtual void visit(const Identifier<GlobalVariable>* o);
  virtual void visit(const Identifier<LocalVariable>* o);
  virtual void visit(const Identifier<MemberVariable>* o);
  virtual void visit(const OverloadedIdentifier<Function>* o);
  virtual void visit(const OverloadedIdentifier<Fiber>* o);
  virtual void visit(const OverloadedIdentifier<MemberFunction>* o);
  virtual void visit(const OverloadedIdentifier<MemberFiber>* o);
  virtual void visit(const OverloadedIdentifier<BinaryOperator>* o);
  virtual void visit(const OverloadedIdentifier<UnaryOperator>* o);

  virtual void visit(const File* o);
  virtual void visit(const GlobalVariable* o);
  virtual void visit(const LocalVariable* o);
  virtual void visit(const MemberVariable* o);
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
  virtual void visit(const WeakType* o);
  virtual void visit(const ClassType* o);
  virtual void visit(const BasicType* o);
  virtual void visit(const GenericType* o);
  virtual void visit(const MemberType* o);
  virtual void visit(const UnknownType* o);
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
  void genTraceLine(const int line);

  /*
   * Generate arguments for function calls with appropriate casts where
   * necessary.
   */
  void genArgs(const Call* o);
  void genLeftArg(const BinaryCall* o);
  void genRightArg(const BinaryCall* o);
  void genSingleArg(const UnaryCall* o);
  void genArg(const Expression* arg, const Type* type);

  /**
   * Output header instead of source?
   */
  bool header;

  /**
   * Are we on the left side of an assignment statement?
   */
  int inAssign;

  /**
   * Are we inside custom pointer code?
   */
  int inPointer;

  /**
   * Are we inside a constructor?
   */
  int inConstructor;

  /**
   * Are we inside the body of a lambda function?
   */
  int inLambda;
};
}

template<class T>
void bi::CppBaseGenerator::genInit(const T* o) {
  if (o->type->isArray()) {
    ArrayType* type = dynamic_cast<ArrayType*>(o->type->canonical());
    assert(type);
    if (!o->brackets->isEmpty()) {
      if (!o->value->isEmpty()) {
        middle(" = ");
        if (o->value->type->isConvertible(*type)) {
          middle(o->value);
        } else {
          middle("libbirch::make_array_and_assign<" << type->single << ">(");
          middle("libbirch::make_frame(" << o->brackets << ')');
          middle(", " << o->value << ')');
        }
      } else {
        middle(" = libbirch::make_array<" << type->single << ">(");
        middle("libbirch::make_frame(" << o->brackets << ')');
        if (!o->args->isEmpty()) {
          middle(", " << o->args);
        }
        middle(')');
      }
    } else if (!o->value->isEmpty()) {
      middle(" = " << o->value);
    }
  } else if (o->type->isClass() || o->type->isWeak()) {
    if (!o->value->isEmpty()) {
      middle(" = " << o->value);
    } else if (!o->type->isWeak()) {
      ++inPointer;
      middle(" = " << o->type << "::create_");
      middle('(' << o->args << ')');
    }
  } else if (!o->value->isEmpty()) {
    middle(" = " << o->value);
  }
}

template<class ObjectType>
void bi::CppBaseGenerator::genTemplateParams(const ObjectType* o) {
  if (o->isGeneric()) {
    start("template<");
    if (!o->isBound()) {
      for (auto iter = o->typeParams->begin(); iter != o->typeParams->end();
          ++iter) {
        if (iter != o->typeParams->begin()) {
          middle(", ");
        }
        middle("class " << *iter);
      }
    }
    finish('>');
  }
}

template<class ObjectType>
void bi::CppBaseGenerator::genTemplateArgs(const ObjectType* o) {
  if (o->isGeneric()) {
    middle('<' << o->typeParams << '>');
  }
}
