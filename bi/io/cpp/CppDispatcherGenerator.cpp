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
  scope = o->scope.get();

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

void bi::CppDispatcherGenerator::visit(const Dispatcher* o) {
  start("template<");
  for (int i = 1; i <= o->types.size(); ++i) {
    if (i != 1) {
      middle(", ");
    }
    middle("class T" << i);
  }
  finish(">");
  if (header) {
    middle("static ");
  }
  start(o->type << " dispatch_" << o->mangled << "_" << o->number << "_(");

  for (int i = 1; i <= o->types.size(); ++i) {
    if (i != 1) {
      middle(", ");
    }
    middle("T" << i << "&& o" << i);
  }
  middle(')');
  if (header) {
    finish(";\n");
  } else {
    finish(" {");
    in();

    /* try functions */
    for (auto iter = o->funcs.begin(); iter != o->funcs.end(); ++iter) {
      *this << *iter;
    }

    /* defer to parent dispatcher */
    Dispatcher* parent = scope->parent(const_cast<Dispatcher*>(o));
    if (parent) {
      line("// dispatch_" << parent->mangled << "_" << parent->number << "_");
    }
    out();
    line("}\n");
  }
}

void bi::CppDispatcherGenerator::visit(const FuncParameter* o) {
  Gatherer<VarParameter> gatherer;
  o->parens->accept(&gatherer);

  start("try { return ");
  if (o->isBinary() && isTranslatable(o->name->str())
      && !o->parens->isRich()) {
    genArg(o->getLeft(), 1);
    middle(' ' << translate(o->name->str()) << ' ');
    genArg(o->getRight(), 2);
  } else if (o->isUnary() && isTranslatable(o->name->str())
      && !o->parens->isRich()) {
    middle(translate(o->name->str()) << ' ');
    genArg(o->getRight(), 1);
  } else {
    middle("bi::" << o->mangled);
    if (o->isConstructor()) {
      middle("<>");
    }
    middle('(');
    int i = 1;
    for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter, ++i) {
      if (iter != gatherer.begin()) {
        middle(", ");
      }
      genArg(*iter, i);
    }
    middle(")");
  }
  finish("; } catch (std::bad_cast e) {}");
}

void bi::CppDispatcherGenerator::visit(const VarParameter* o) {
  middle(o->name);
}

void bi::CppDispatcherGenerator::genArg(const Expression* o, const int i) {
  middle("bi::cast<");
  if (!o->type->assignable) {
    middle("const ");
  }
  middle(o->type << "&>(o" << i << ')');
}
