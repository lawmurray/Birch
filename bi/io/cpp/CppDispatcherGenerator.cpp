/**
 * @file
 */
#include "bi/io/cpp/CppDispatcherGenerator.hpp"

#include "bi/io/cpp/CppParameterGenerator.hpp"
#include "bi/io/cpp/misc.hpp"
#include "bi/visitor/DispatchGatherer.hpp"
#include "bi/visitor/Gatherer.hpp"

bi::CppDispatcherGenerator::CppDispatcherGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppDispatcherGenerator::visit(const File* o) {
  DispatchGatherer gatherer;
  o->accept(&gatherer);

  header = true;
  for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
    if (*(*iter)->name != "<-") {
      *this << *iter;
    }
  }

  header = false;
  for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
    if (*(*iter)->name != "<-") {
      *this << *iter;
    }
  }
}

void bi::CppDispatcherGenerator::visit(const VarParameter* o) {
  middle(o->name);
}

void bi::CppDispatcherGenerator::visit(const FuncParameter* o) {
  Gatherer<VarParameter> gatherer;
  o->parens->accept(&gatherer);

  int i;
  start("template<");
  for (i = 1; i <= gatherer.size(); ++i) {
    if (i != 1) {
      middle(", ");
    }
    middle("class T" << i);
  }
  finish(">");
  if (header) {
    middle("static ");
  }
  start(o->type << " dispatch_" << o->number << "_(");

  i = 1;
  for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter, ++i) {
    if (iter != gatherer.begin()) {
      middle(", ");
    }
    middle("T" << i << "&& " << (*iter)->name);
  }
  middle(')');
  if (header) {
    finish(";\n");
  } else {
    finish(" {");
    in();
    start("return ");

    /* definite call */
    if (o->isBinary() && isTranslatable(o->name->str())
        && !o->parens->isRich()) {
      genArg(o->getLeft());
      middle(' ' << translate(o->name->str()) << ' ');
      genArg(o->getRight());
    } else if (o->isUnary() && isTranslatable(o->name->str())
        && !o->parens->isRich()) {
      middle(translate(o->name->str()) << ' ');
      genArg(o->getRight());
    } else {
      middle("bi::" << o->mangled);
      if (o->isConstructor()) {
        middle("<>");
      }
      middle('(');
      for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
        if (iter != gatherer.begin()) {
          middle(", ");
        }
        genArg(*iter);
      }
      middle(")");
    }
    finish(';');
    out();
    finish("}\n");
  }
}

void bi::CppDispatcherGenerator::genArg(const Expression* o) {
  middle("bi::cast<");
  if (!o->type->assignable) {
    middle("const ");
  }
  middle(o->type << "&>(" << o << ')');
}
