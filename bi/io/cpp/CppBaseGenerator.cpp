/**
 * @file
 */
#include "bi/io/cpp/CppBaseGenerator.hpp"

#include "bi/io/cpp/misc.hpp"

bi::CppBaseGenerator::CppBaseGenerator(std::ostream& base, const int level,
    const bool header) :
    indentable_ostream(base, level),
    header(header) {
  //
}

void bi::CppBaseGenerator::visit(const BooleanLiteral* o) {
  *this << "bi::make_bool(" << o->str << ')';
}

void bi::CppBaseGenerator::visit(const IntegerLiteral* o) {
  *this << "bi::make_int(" << o->str << ')';
}

void bi::CppBaseGenerator::visit(const RealLiteral* o) {
  *this << "bi::make_real(" << o->str << ')';
}

void bi::CppBaseGenerator::visit(const StringLiteral* o) {
  *this << "bi::make_string(" << o->str << ')';
}

void bi::CppBaseGenerator::visit(const Name* o) {
  *this << o->str();
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

void bi::CppBaseGenerator::visit(const Range* o) {
  middle("bi::make_range(" << o->left << ", " << o->right << ')');
}

void bi::CppBaseGenerator::visit(const This* o) {
  middle("*this");
}

void bi::CppBaseGenerator::visit(const RandomInit* o) {
  finish(o->left << ".init(" << o->right << ",");
  in();
  in();
  line("[](" << o->left->type << "& rv) {");
  in();
  line("pull_(rv.x, rv.m);");
  out();
  line("}, [&]() { ");
  in();
  line(o->push << ';');
  out();
  start("})");
  out();
  out();
}

void bi::CppBaseGenerator::visit(const Member* o) {
  const This* left = dynamic_cast<const This*>(o->left.get());
  if (left) {
    // tidier this way
    middle("this->" << o->right);
  } else {
    middle(o->left << '.' << o->right);
  }
}

void bi::CppBaseGenerator::visit(const VarReference* o) {
  middle(o->name);
}

void bi::CppBaseGenerator::visit(const FuncReference* o) {
  if (*o->name == "<-") {
    //if (*o->getLeft()->type <= *o->getRight()->type) {
      middle(o->getLeft() << " = " <<  o->getRight());
    //} else {
    //  middle("bi::" << o->target->mangled);
    //  middle('(' << o->getLeft() << ", " << o->getRight() << ')');
    //}
  } else if (o->isBinary() && isTranslatable(o->name->str())
      && !o->target->parens->isRich()) {
    //if (arg1->isPrimary()) {
    middle(o->getLeft());
    middle(' ' << translate(o->name->str()) << ' ');
    middle(o->getRight());
  } else if (o->isUnary() && isTranslatable(o->name->str())
      && !o->target->parens->isRich()) {
    assert(o->args.size() == 1);
    auto iter = o->args.begin();
    middle(translate(o->name->str()) << ' ' << *iter);
  } else {
    middle("bi::");
    //middle("function::");
    middle(o->target->mangled);
    if (o->isConstructor()) {
      middle("<>");
    }
    middle('(');
    for (auto iter = o->args.begin(); iter != o->args.end(); ++iter) {
      if (iter != o->args.begin()) {
        middle(", ");
      }
      middle(*iter);
    }
    middle(')');
  }
}

void bi::CppBaseGenerator::visit(const ModelReference* o) {
  if (o->count() > 0) {
    middle("bi::Array<bi::model::" << o->name << "<bi::HeapGroup>,");
    middle("typename bi::DefaultFrame<" << o->count() << ">::type>");
  } else {
    middle("bi::model::" << o->name << "<>");
  }
}

void bi::CppBaseGenerator::visit(const VarParameter* o) {
  middle(o->type << ' ' << o->name);
  if (!o->parens->isEmpty() || o->type->count() > 0) {
    middle('(');
  }
  if (!o->parens->isEmpty()) {
    middle(o->parens->strip());
    if (o->type->count() > 0) {
      middle(", ");
    }
  }
  if (o->type->count() > 0) {
    ModelReference* type = dynamic_cast<ModelReference*>(o->type.get());
    assert(type);
    middle("make_frame(" << type->brackets << ")");
    if (!o->value->isEmpty()) {
      middle(", " << o->value->strip());
    }
  }
  if (!o->parens->isEmpty() || o->type->count() > 0) {
    middle(')');
  }
  if (!o->value->isEmpty()) {
    middle(" = " << o->value);
  }
}

void bi::CppBaseGenerator::visit(const ExpressionStatement* o) {
  line(o->single << ';');
}

void bi::CppBaseGenerator::visit(const Conditional* o) {
  line("if " << o->cond << " {");
  in();
  *this << o->braces;
  out();
  if (o->falseBraces->isEmpty()) {
    line("} else {");
    in();
    *this << o->falseBraces;
    out();
  }
  line("}");
}

void bi::CppBaseGenerator::visit(const Loop* o) {
  line("while " << o->cond << " {");
  in();
  *this << o->braces;
  out();
  line("}");
}

void bi::CppBaseGenerator::visit(const Raw* o) {
  if ((header && o->name->str().compare("hpp") == 0)
      || (!header && o->name->str().compare("cpp") == 0)) {
    *this << o->raw;
    if (!std::isspace(o->raw.back())) {
      *this << ' ';
    }
  }
}

void bi::CppBaseGenerator::visit(const EmptyType* o) {
  middle("void");
}

void bi::CppBaseGenerator::visit(const ParenthesesType* o) {
  if (dynamic_cast<TypeList*>(o->single->strip())) {
    middle("std::tuple<" << o->single->strip() << ">");
  } else {
    middle(o->single);
  }
}

void bi::CppBaseGenerator::visit(const RandomType* o) {
  middle("bi::Random<");
  middle(o->left << ',' << o->right);
  middle(">");
}
