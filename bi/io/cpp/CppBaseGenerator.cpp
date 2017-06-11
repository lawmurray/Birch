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

void bi::CppBaseGenerator::visit(const BooleanLiteral* o) {
  middle(o->str);
}

void bi::CppBaseGenerator::visit(const IntegerLiteral* o) {
  middle(o->str);
}

void bi::CppBaseGenerator::visit(const RealLiteral* o) {
  middle(o->str);
}

void bi::CppBaseGenerator::visit(const StringLiteral* o) {
  middle("std::string(" << o->str << ')');
}

void bi::CppBaseGenerator::visit(const ExpressionList* o) {
  middle(o->head);
  if (o->tail) {
    middle(", " << o->tail);
  }
}

void bi::CppBaseGenerator::visit(const StatementList* o) {
  middle(o->head);
  if (o->tail) {
    middle(o->tail);
  }
}

void bi::CppBaseGenerator::visit(const TypeList* o) {
  middle(o->head);
  Type* tail = o->tail.get();
  TypeList* list = dynamic_cast<TypeList*>(tail);
  while (list) {
    middle(',' << list->head);
    tail = list->tail.get();
    list = dynamic_cast<TypeList*>(tail);
  }
  middle(',' << tail);
}

void bi::CppBaseGenerator::visit(const ParenthesesExpression* o) {
  if (o->single->tupleSize() > 1) {
    middle("std::make_tuple");
  }
  middle('(' << o->single << ')');
}

void bi::CppBaseGenerator::visit(const BracesExpression* o) {
  //finish('{');
  //in();
  *this << o->single;
  //out();
  //start('}');
}

void bi::CppBaseGenerator::visit(const BracketsExpression* o) {
  middle(o->single << "(make_view(" << o->brackets << "))");
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

void bi::CppBaseGenerator::visit(const Super* o) {
  middle("dynamic_cast<super_type*>(this)");
}

void bi::CppBaseGenerator::visit(const This* o) {
  middle("this");
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

void bi::CppBaseGenerator::visit(const VarReference* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const FuncReference* o) {
  if (o->isAssign() && *o->name == "<-" && !o->target) {
    if (o->getLeft()->type->isClass() && !o->getRight()->type->isClass()) {
      middle("*(" << o->getLeft() << ')');
    } else {
      middle(o->getLeft());
    }
    middle(" = ");
    if (!o->getLeft()->type->isClass() && o->getRight()->type->isClass()) {
      middle("*(" << o->getRight() << ')');
    } else {
      middle(o->getRight());
    }
  } else if (o->isBinary() && isTranslatable(o->name->str())) {
    genArg(o->getLeft(), o->target->getLeft());
    middle(' ' << o->name << ' ');
    genArg(o->getRight(), o->target->getRight());
  } else if (o->isUnary() && isTranslatable(o->name->str())) {
    middle(o->name);
    genArg(o->getRight(), o->target->getRight());
  } else {
    if (!o->isMember() && !o->isLambda() && o->name->str() != "assert") {  // special exception for now
      middle("bi::");
      if (!o->isOperator()) {
        middle("func::");
      }
    }
    if (o->isCoroutine() && o->isLambda()) {
      middle("(*" << internalise(o->name->str()) << ')');
    } else {
      middle(internalise(o->name->str()));
    }
    middle('(');
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
}

void bi::CppBaseGenerator::visit(const BinaryReference* o) {
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

void bi::CppBaseGenerator::visit(const VarParameter* o) {
  middle(o->type << ' ' << o->name);
  if (o->type->isClass()) {
    TypeReference* type = dynamic_cast<TypeReference*>(o->type->strip());
    assert(type);
    middle(
        " = new (GC_MALLOC(sizeof(bi::type::" << type->name << "))) bi::type::" << type->name << '(');
  } else if (!o->parens->isEmpty() || o->type->count() > 0) {
    middle('(');
  }
  if (o->type->count() > 0) {
    BracketsType* type = dynamic_cast<BracketsType*>(o->type->strip());
    assert(type);
    middle("make_frame(" << type->brackets << ")");
  }
  if (!o->parens->isEmpty()) {
    if (o->type->count() > 0) {
      middle(", ");
    }
    middle(o->parens->strip());
  }
  if (o->type->isClass() || !o->parens->isEmpty() || o->type->count() > 0) {
    middle(')');
  }
  if (!o->value->isEmpty()) {
    if (o->type->isClass()) {
      ///@todo How to handle this case?
      assert(false);
    } else {
      middle(" = " << o->value);
    }
  }
  finish(';');
}

void bi::CppBaseGenerator::visit(const FuncParameter* o) {
  if (o->isLambda()) {
    middle("[&](");
    CppParameterGenerator auxParameter(base, level, header);
    auxParameter << o->parens;
    finish(") {");
    in();
    *this << o->braces;
    out();
    start("}");
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

void bi::CppBaseGenerator::visit(const TypeReference* o) {
  if (o->isBuiltin()) {
    genBuiltin(o);
  } else if (o->isClass()) {
    middle("bi::reloc_ptr<bi::type::" << o->name << ">");
  } else {
    middle("bi::type::" << o->name);
  }
}

void bi::CppBaseGenerator::visit(const EmptyType* o) {
  middle("void");
}

void bi::CppBaseGenerator::visit(const BracketsType* o) {
  middle("bi::DefaultArray<" << o->single << ',' << o->count() << '>');
}

void bi::CppBaseGenerator::visit(const ParenthesesType* o) {
  if (dynamic_cast<TypeList*>(o->single->strip())) {
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
  middle("bi::reloc_ptr<bi::Coroutine<" << o->type << ">>");
}

void bi::CppBaseGenerator::genCapture(const Expression* o) {
  /* for lambda, capture assignable variables by reference, others by value */
  Gatherer<VarReference> gatherer;
  o->accept(&gatherer);
  std::unordered_set<std::string> done;

  middle("[=");
  for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
    const VarReference* ref = *iter;
    if (done.find(ref->name->str()) == done.end()) {
      done.insert(ref->name->str());
    }
  }
  middle(']');
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
