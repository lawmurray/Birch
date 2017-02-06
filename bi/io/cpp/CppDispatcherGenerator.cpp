/**
 * @file
 */
#include "bi/io/cpp/CppDispatcherGenerator.hpp"

#include "bi/io/cpp/CppParameterGenerator.hpp"
#include "bi/io/cpp/misc.hpp"
#include "bi/visitor/DispatchGatherer.hpp"
#include "bi/visitor/Gatherer.hpp"

bi::CppDispatcherGenerator::CppDispatcherGenerator(Scope* scope,
    std::ostream& base, const int level, const bool header) :
    CppBaseGenerator(base, level, header),
    scope(scope) {
  //
}

void bi::CppDispatcherGenerator::visit(const File* o) {
  DispatchGatherer gatherer(o->scope.get());
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
    start("try { return bi::");
    if ((o->isBinary() || o->isUnary()) && isTranslatable(o->name->str())
        && !o->parens->isRich()) {
      middle("operator" << translate(o->name->str()));
    } else {
      middle(o->mangled);
    }
    middle('(');

    for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
      if (iter != gatherer.begin()) {
        middle(", ");
      }
      middle("bi::cast<");
      if (!(*iter)->type->assignable) {
        middle("const ");
      }
      middle((*iter)->type << "&>(" << (*iter)->name << ')');
    }
    finish("); } catch (std::bad_cast) {}");

    std::list<FuncParameter*> parents;
    scope->parents(const_cast<FuncParameter*>(o), parents);
    for (auto iter = parents.begin(); iter != parents.end(); ++iter) {
      start("try { return ");
      bool result = const_cast<FuncParameter*>(o)->definitely(**iter);  // needed to capture arguments
      assert(result);
      middle("dispatch_" << (*iter)->number << '_');
      genArgs(const_cast<FuncParameter*>(o), *iter);
      finish("; } catch (std::bad_cast) {}");
    }
    line("throw std::bad_cast();");
    out();
    finish("}\n");
  }
}
