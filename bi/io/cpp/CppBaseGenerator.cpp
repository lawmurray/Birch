/**
 * @file
 */
#include "bi/io/cpp/CppBaseGenerator.hpp"

#include "bi/io/cpp/CppOutputGenerator.hpp"
#include "bi/io/cpp/CppParameterGenerator.hpp"
#include "bi/io/cpp/CppReturnGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"
#include "bi/primitive/encode.hpp"

#include <unordered_set>

bi::CppBaseGenerator::CppBaseGenerator(std::ostream& base, const int level,
    const bool header) :
    indentable_ostream(base, level),
    header(header),
    inDelay(0),
    inLambda(0),
    inVariant(0),
    inReturn(0) {
  //
}

void bi::CppBaseGenerator::visit(const Name* o) {
  middle(internalise(o->str()));
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
  middle(o->str);
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
  middle(o->single << "(bi::make_view(" << o->brackets << "))");
}

void bi::CppBaseGenerator::visit(const Index* o) {
  middle("bi::make_index(" << o->single << ')');
}

void bi::CppBaseGenerator::visit(const Range* o) {
  middle("bi::make_range(" << o->left << ", " << o->right << ')');
}

void bi::CppBaseGenerator::visit(const This* o) {
  middle("*nonconst(this)");
}

void bi::CppBaseGenerator::visit(const Member* o) {
  const This* left = dynamic_cast<const This*>(o->left.get());
  if (left) {
    // tidier this way
    middle("nonconst(this)->");
  } else {
    middle(o->left);
    if (o->left->type->polymorphic && o->left->isMember()) {
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
  if (o->dispatcher) {
    middle("dispatch_" << o->dispatcher->name << '_');
    middle(o->dispatcher->number << "_(" << o->parens << ')');
  } else if (o->isAssign() && *o->name == "<-") {
    middle(o->getLeft() << " = ");
    if (o->getLeft()->isMember() && o->getLeft()->type->polymorphic) {
      middle("new " << o->getRight()->type << '(' << o->getRight() <<')');
    } else {
      middle(o->getRight());
    }
  } else if (o->isBinary() && isTranslatable(o->name->str())) {
    middle(o->getLeft() << ' ' << o->name << ' ' << o->getRight());
  } else if (o->isUnary() && isTranslatable(o->name->str())) {
    middle(o->name << o->getRight());
  } else {
    middle(o->name << '(' << o->parens << ')');
  }
}

void bi::CppBaseGenerator::visit(const VarParameter* o) {
  middle(o->type << ' ' << o->name);
//  if (!o->parens->isEmpty() || o->type->count() > 0) {
//    middle('(');
//  }
//  if (!o->parens->isEmpty()) {
//    middle(o->parens->strip());
//    if (o->type->count() > 0) {
//      middle(", ");
//    }
//  }
//  if (o->type->count() > 0) {
//    BracketsType* type = dynamic_cast<BracketsType*>(o->type.get());
//    assert(type);
//    middle("make_frame(" << type->brackets << ")");
//    if (!o->value->isEmpty()) {
//      middle(", " << o->value->strip());
//    }
//  }
//  if (!o->parens->isEmpty() || o->type->count() > 0) {
//    middle(')');
//  }
  if (!o->value->isEmpty()) {
    middle(" = " << o->value);
  }
}

void bi::CppBaseGenerator::visit(const FuncParameter* o) {
  if (o->isLambda()) {
    middle("[&](");
    CppParameterGenerator auxParameter(base, level, header);
    auxParameter << o->parens;
    finish(") {");
    in();

    /* output parameters */
    CppOutputGenerator auxOutput(base, level, false);
    auxOutput << o;

    /* body */
    *this << o->braces;

    /* return statement */
    if (!o->result->isEmpty()) {
      CppReturnGenerator auxReturn(base, level, false);
      auxReturn << o;
    }

    out();
    start("}");
  }
}

void bi::CppBaseGenerator::visit(const VarDeclaration* o) {
  line(o->param << ';');
}

void bi::CppBaseGenerator::visit(const ExpressionStatement* o) {
  line(o->single << ';');
}

void bi::CppBaseGenerator::visit(const Conditional* o) {
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

void bi::CppBaseGenerator::visit(const Loop* o) {
  line("while (" << o->cond << ") {");
  in();
  *this << o->braces;
  out();
  line("}");
}

void bi::CppBaseGenerator::visit(const Raw* o) {
  if ((header && *o->name == "hpp") || (!header && *o->name == "cpp")) {
    *this << o->raw;
    if (!std::isspace(o->raw.back())) {
      *this << ' ';
    }
  }
}

void bi::CppBaseGenerator::visit(const ModelReference* o) {
  if (o->isBuiltin()) {
    genBuiltin(o);
  } else {
    middle("bi::model::" << o->name << "<>");
  }
}

void bi::CppBaseGenerator::visit(const EmptyType* o) {
  if (inVariant) {
    middle("boost::blank");
  } else {
    middle("void");
  }
}

void bi::CppBaseGenerator::visit(const BracketsType* o) {
  if (!o->assignable) {
    middle("const ");
  }
  middle("DefaultArray<" << o->single << ',' << o->count() << '>');
}

void bi::CppBaseGenerator::visit(const ParenthesesType* o) {
  if (dynamic_cast<TypeList*>(o->single->strip())) {
    middle("std::tuple<" << o->single->strip() << ">");
  } else {
    middle(o->single);
  }
}

void bi::CppBaseGenerator::visit(const DelayType* o) {
  ++inDelay;
  middle("bi::Delay<" << o->left << ',' << o->right << '>');
  --inDelay;
}

void bi::CppBaseGenerator::visit(const LambdaType* o) {
  ++inLambda;
  middle("bi::Lambda<" << o->result << '(');
  CppParameterGenerator auxParameter(base, level, header);
  auxParameter << o->parens;
  middle(")>");
  --inLambda;
}

void bi::CppBaseGenerator::visit(const VariantType* o) {
  ++inVariant;
  middle("bi::Variant<" << o->definite);
  for (auto iter = o->possibles.begin(); iter != o->possibles.end(); ++iter) {
    middle(',');
    middle(*iter);
  }
  middle(">");
  --inVariant;
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
      if (ref->type->isDelay()) {
        middle(", &" << ref->name);
      }
      done.insert(ref->name->str());
    }
  }
  middle(']');
}

void bi::CppBaseGenerator::genBuiltin(const ModelReference* o) {
  /* pre-condition */
  assert(o->isBuiltin());

  if (*o->name == "Boolean") {
    middle("unsigned char");
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
