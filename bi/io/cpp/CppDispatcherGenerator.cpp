/**
 * @file
 */
#include "bi/io/cpp/CppDispatcherGenerator.hpp"

#include "bi/io/cpp/CppParameterGenerator.hpp"
#include "bi/io/cpp/misc.hpp"
#include "bi/visitor/Gatherer.hpp"

bi::CppDispatcherGenerator::CppDispatcherGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppDispatcherGenerator::visit(const File* o) {
  header = true;
  for (auto iter = o->scope->dispatchers.begin();
      iter != o->scope->dispatchers.end(); ++iter) {
    *this << iter->second;
  }

  header = false;
  for (auto iter = o->scope->dispatchers.begin();
      iter != o->scope->dispatchers.end(); ++iter) {
    *this << iter->second;
  }
}

void bi::CppDispatcherGenerator::visit(const Dispatcher* o) {
  Gatherer<VarParameter> gatherer;
  o->parens->accept(&gatherer);

  start("template<");
  int i = 1;
  for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter, ++i) {
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

  i = 1;
  for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter, ++i) {
    if (i != 1) {
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

    /* try functions, in topological order from most specific */
    for (auto iter = o->funcs.rbegin(); iter != o->funcs.rend(); ++iter) {
      *this << *iter;
    }

    /* defer to parent dispatcher */
    if (o->parent) {
      assert(o->parens->possibly(*o->parent->parens));
      o->parens->possibly(*o->parent->parens);

      Gatherer<VarParameter> gatherer;
      o->parent->parens->accept(&gatherer);

      start(
          "return dispatch_" << o->parent->mangled << "_" << o->parent->number << "_(");
      for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
        if (iter != gatherer.begin()) {
          middle(", ");
        }
        genArg((*iter)->arg, i);
      }
      finish(");");
    } else {
      line("throw std::bad_cast();");
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
