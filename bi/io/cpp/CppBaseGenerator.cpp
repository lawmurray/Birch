/**
 * @file
 */
#include "bi/io/cpp/CppBaseGenerator.hpp"

#include "bi/io/cpp/CppParameterGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"
#include "bi/primitive/encode.hpp"

#include <unordered_set>

bi::CppBaseGenerator::CppBaseGenerator(std::ostream& base, const int level,
    const bool header) :
    indentable_ostream(base, level),
    header(header),
    inReturn(0) {
  //
}

void bi::CppBaseGenerator::visit(const Name* o) {
  middle(o->str());
}

void bi::CppBaseGenerator::visit(const List<Expression>* o) {
  middle(o->head);
  if (o->tail) {
    middle(", " << o->tail);
  }
}

void bi::CppBaseGenerator::visit(const Literal<bool>* o) {
  middle(o->str);
}

void bi::CppBaseGenerator::visit(const Literal<int64_t>* o) {
  middle(o->str);
}

void bi::CppBaseGenerator::visit(const Literal<double>* o) {
  middle(o->str);
}

void bi::CppBaseGenerator::visit(const Literal<const char*>* o) {
  middle("std::string(" << o->str << ')');
}

void bi::CppBaseGenerator::visit(const ParenthesesExpression* o) {
  if (o->single->tupleSize() > 1) {
    middle("std::make_tuple");
  }
  middle('(' << o->single << ')');
}

void bi::CppBaseGenerator::visit(const BracesExpression* o) {
  *this << o->single;
}

void bi::CppBaseGenerator::visit(const BracketsExpression* o) {
  middle(o->single << "(make_view(" << o->brackets << "))");
}

void bi::CppBaseGenerator::visit(const LambdaFunction* o) {
  middle("[&](");
  CppParameterGenerator auxParameter(base, level, header);
  auxParameter << o->parens;
  finish(") {");
  in();
  *this << o->braces;
  out();
  start("}");
}

void bi::CppBaseGenerator::visit(const Span* o) {
  middle("make_span(" << o->single << ')');
}

void bi::CppBaseGenerator::visit(const Index* o) {
  middle("make_index(" << o->single << " - 1)");
}

void bi::CppBaseGenerator::visit(const Range* o) {
  middle("make_range(" << o->left << " - 1, " << o->right << " - 1)");
}

void bi::CppBaseGenerator::visit(const Member* o) {
  const This* leftThis = dynamic_cast<const This*>(o->left.get());
  const Super* leftSuper = dynamic_cast<const Super*>(o->left.get());
  if (leftThis) {
    // tidier this way
    middle("this->");
  } else if (leftSuper) {
    // tidier this way
    middle("super_type::");
  } else {
    middle(o->left);
    if (o->left->type->isClass()) {
      middle("->");
    } else {
      middle('.');
    }
  }
  middle(o->right);
}

void bi::CppBaseGenerator::visit(const Super* o) {
  middle("dynamic_cast<super_type*>(this)");
}

void bi::CppBaseGenerator::visit(const This* o) {
  middle("this");
}

void bi::CppBaseGenerator::visit(const Identifier<Parameter>* o) {
  middle(internalise(o->name->str()));
}

void bi::CppBaseGenerator::visit(const Identifier<GlobalVariable>* o) {
  middle(internalise(o->name->str()));
}

void bi::CppBaseGenerator::visit(const Identifier<LocalVariable>* o) {
  middle(internalise(o->name->str()));
}

void bi::CppBaseGenerator::visit(const Identifier<MemberVariable>* o) {
  middle(internalise(o->name->str()));
}

void bi::CppBaseGenerator::visit(const Identifier<Function>* o) {
  middle("bi::func::" << internalise(o->name->str()) << '(');
  auto arg = o->parens->begin();
  auto param = o->target->parens->begin();
  while (arg != o->parens->end() && param != o->target->parens->end()) {
    if (arg != o->parens->begin()) {
      middle(", ");
    }
    genArg(*arg, *param);
    ++arg;
    ++param;
  }
  assert(arg == o->parens->end());
  assert(param == o->target->parens->end());
  middle(')');
}

void bi::CppBaseGenerator::visit(const Identifier<Coroutine>* o) {
  middle("bi::func::" << internalise(o->name->str()) << '(');
  auto arg = o->parens->begin();
  auto param = o->target->parens->begin();
  while (arg != o->parens->end() && param != o->target->parens->end()) {
    if (arg != o->parens->begin()) {
      middle(", ");
    }
    genArg(*arg, *param);
    ++arg;
    ++param;
  }
  assert(arg == o->parens->end());
  assert(param == o->target->parens->end());
  middle(')');
}

void bi::CppBaseGenerator::visit(const Identifier<MemberFunction>* o) {
  middle(internalise(o->name->str()) << '(');
  auto arg = o->parens->begin();
  auto param = o->target->parens->begin();
  while (arg != o->parens->end() && param != o->target->parens->end()) {
    if (arg != o->parens->begin()) {
      middle(", ");
    }
    genArg(*arg, *param);
    ++arg;
    ++param;
  }
  assert(arg == o->parens->end());
  assert(param == o->target->parens->end());
  middle(')');
}

void bi::CppBaseGenerator::visit(const Identifier<BinaryOperator>* o) {
  if (isTranslatable(o->name->str())) {
    /* can use as raw C++ operator */
    genArg(o->left.get(), o->target->left.get());
    middle(' ' << o->name << ' ');
    genArg(o->right.get(), o->target->right.get());
  } else {
    /* must use as function */
    middle("bi::" << internalise(o->name->str()) << '(');
    genArg(o->left.get(), o->target->left.get());
    middle(", ");
    genArg(o->right.get(), o->target->right.get());
    middle(')');
  }
}

void bi::CppBaseGenerator::visit(const Identifier<UnaryOperator>* o) {
  if (isTranslatable(o->name->str())) {
    /* can use as raw C++ operator */
    middle(' ' << o->name);
    genArg(o->single.get(), o->target->single.get());
  } else {
    /* must use as function */
    middle("bi::" << internalise(o->name->str()) << '(');
    genArg(o->single.get(), o->target->single.get());
    middle(')');
  }
}

void bi::CppBaseGenerator::visit(const Parameter* o) {
  middle(o->type << ' ' << o->name);
}

void bi::CppBaseGenerator::visit(const GlobalVariable* o) {
  middle(o->type << ' ' << o->name);
  if (o->type->isClass()) {
    TypeReference* type = dynamic_cast<TypeReference*>(o->type->strip());
    assert(type);
    middle(
        " = new (GC_MALLOC(sizeof(bi::type::" << type->name << "))) bi::type::" << type->name << "()");
  }
  if (o->type->count() > 0) {
    BracketsType* type = dynamic_cast<BracketsType*>(o->type->strip());
    assert(type);
    middle("(make_frame(" << type->brackets << "))");
  }
}

void bi::CppBaseGenerator::visit(const LocalVariable* o) {
  middle(o->type << ' ' << o->name);
  if (o->type->isClass()) {
    TypeReference* type = dynamic_cast<TypeReference*>(o->type->strip());
    assert(type);
    middle(
        " = new (GC_MALLOC(sizeof(bi::type::" << type->name << "))) bi::type::" << type->name << "()");
  }
  if (o->type->count() > 0) {
    BracketsType* type = dynamic_cast<BracketsType*>(o->type->strip());
    assert(type);
    middle("(make_frame(" << type->brackets << "))");
  }
}

void bi::CppBaseGenerator::visit(const MemberVariable* o) {
  middle(o->type << ' ' << o->name);
}

void bi::CppBaseGenerator::visit(const List<Statement>* o) {
  middle(o->head);
  if (o->tail) {
    middle(o->tail);
  }
}

void bi::CppBaseGenerator::visit(const Assignment* o) {
  if (o->left->type->isClass() && !o->right->type->isClass()) {
    middle("*(" << o->left << ')');
  } else {
    middle(o->left);
  }
  middle(" = ");
  if (!o->left->type->isClass() && o->right->type->isClass()) {
    middle("*(" << o->right << ')');
  } else {
    middle(o->right);
  }
}

void bi::CppBaseGenerator::visit(const ExpressionStatement* o) {
  line(o->single << ';');
}

void bi::CppBaseGenerator::visit(const If* o) {
  line("if (" << o->cond << ") {");
  in();
  *this << o->braces;
  out();
  if (!o->falseBraces->isEmpty()) {
    line("} else {");
    in();
    *this << o->falseBraces;
    out();
  }
  line("}");
}

void bi::CppBaseGenerator::visit(const For* o) {
  ///@todo May need to be more sophisticated to accommodate arbitrary types
  line(
      "for (" << o->index << " = " << o->from << "; " << o->index << " <= " << o->to << "; ++" << o->index << ") {");
  in();
  *this << o->braces;
  out();
  line("}");
}

void bi::CppBaseGenerator::visit(const While* o) {
  line("while (" << o->cond << ") {");
  in();
  *this << o->braces;
  out();
  line("}");
}

void bi::CppBaseGenerator::visit(const Return* o) {
  line("return " << o->single << ';');
}

void bi::CppBaseGenerator::visit(const Raw* o) {
  if ((header && *o->name == "hpp") || (!header && *o->name == "cpp")) {
    *this << o->raw;
    if (!std::isspace(o->raw.back())) {
      *this << ' ';
    }
  }
}

void bi::CppBaseGenerator::visit(const EmptyType* o) {
  middle("void");
}

void bi::CppBaseGenerator::visit(const List<Type>* o) {
  middle(o->head);
  Type* tail = o->tail.get();
  List<Type>* list = dynamic_cast<List<Type>*>(tail);
  while (list) {
    middle(',' << list->head);
    tail = list->tail.get();
    list = dynamic_cast<List<Type>*>(tail);
  }
  middle(',' << tail);
}

void bi::CppBaseGenerator::visit(const TypeReference* o) {
  if (o->isBuiltin()) {
    genBuiltin(o);
  } else if (o->isClass()) {
    middle("bi::Pointer<bi::type::" << o->name << ">");
  } else {
    middle("bi::type::" << o->name);
  }
}

void bi::CppBaseGenerator::visit(const BracketsType* o) {
  middle("bi::DefaultArray<" << o->single << ',' << o->count() << '>');
}

void bi::CppBaseGenerator::visit(const ParenthesesType* o) {
  if (dynamic_cast<List<Type>*>(o->single->strip())) {
    middle("std::tuple<" << o->single->strip() << ">");
  } else {
    middle(o->single);
  }
}

void bi::CppBaseGenerator::visit(const FunctionType* o) {
  middle("std::function<" << o->type << '(');
  CppParameterGenerator auxParameter(base, level, header);
  auxParameter << o->parens;
  middle(")>");
}

void bi::CppBaseGenerator::visit(const CoroutineType* o) {
  middle("bi::Pointer<bi::Coroutine<" << o->type << ">>");
}

void bi::CppBaseGenerator::genBuiltin(const TypeReference* o) {
  /* pre-condition */
  assert(o->isBuiltin());

  if (*o->name == "Boolean") {
    middle("bool");
  } else if (*o->name == "Real64" || *o->name == "Real") {
    middle("double");
  } else if (*o->name == "Real32") {
    middle("float");
  } else if (*o->name == "Integer64" || *o->name == "Integer") {
    middle("int64_t");
  } else if (*o->name == "Integer32") {
    middle("int32_t");
  } else if (*o->name == "String") {
    middle("std::string");
  } else {
    assert(false);
  }
}

void bi::CppBaseGenerator::genArg(const Expression* arg,
    const Expression* param) {
  if (arg->type->isClass() && !param->type->isClass()) {
    middle("*(" << arg << ')');
  } else {
    middle(arg);
  }
}
