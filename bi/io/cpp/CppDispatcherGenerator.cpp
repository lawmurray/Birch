/**
 * @file
 */
#include "bi/io/cpp/CppDispatcherGenerator.hpp"

#include "bi/io/cpp/CppTemplateParameterGenerator.hpp"
#include "bi/io/cpp/CppParameterGenerator.hpp"
#include "bi/visitor/DispatchGatherer.hpp"
#include "bi/visitor/Gatherer.hpp"

bi::CppDispatcherGenerator::CppDispatcherGenerator(Scope* scope, std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header), scope(scope) {
  //
}

void bi::CppDispatcherGenerator::visit(const File* o) {
  DispatchGatherer gatherer;
  o->accept(&gatherer);

  for (auto iter = gatherer.gathered.begin(); iter != gatherer.gathered.end();
      ++iter) {
    if (*(*iter)->name != "<-") {
      *this << *iter;
    }
  }
}

void bi::CppDispatcherGenerator::visit(const ModelReference* o) {
  if (!o->assignable) {
    middle("const ");
  }
  if (o->count() > 0) {
    middle("DefaultArray<" << o->name << "<HeapGroup>," << o->count() << '>');
  } else {
    middle("bi::model::" << o->name << "<>");
  }
}

void bi::CppDispatcherGenerator::visit(const VarParameter* o) {
  middle(o->name);
}

void bi::CppDispatcherGenerator::visit(const FuncParameter* o) {
  /* dispatchers are created as lambda functions so that parameters can use
   * `auto` as their type; up to C++14 standard functions cannot use this */
  start("static auto dispatch_" << o->number << "_ = [](");
  for (auto iter = o->inputs.begin(); iter != o->inputs.end(); ++iter) {
    if (iter != o->inputs.begin()) {
      middle(", ");
    }
    middle("auto&& " << (*iter)->name);
  }
  finish(") -> " << o->type << " {");
  in();
  start("try { return bi::" << o->mangled << '(');
  for (auto iter = o->inputs.begin(); iter != o->inputs.end(); ++iter) {
    if (iter != o->inputs.begin()) {
      middle(", ");
    }
    middle("dynamic_cast<" << (*iter)->type << "&>(" << (*iter)->name << ')');
  }
  finish("); } catch (std::bad_cast) {}");

  std::list<FuncParameter*> parents;
  scope->parents(const_cast<FuncParameter*>(o), parents);
  for (auto iter = parents.begin(); iter != parents.end(); ++iter) {
    start("try { return ");
    possibly result = *const_cast<FuncParameter*>(o) <= **iter;  // needed to capture arguments
    assert(result != untrue);
    if (result == possible) {
      middle("dispatch_" << o->number << "_");
    } else {
      middle("bi::" << o->mangled);
    }
    middle('(');
    Gatherer<VarParameter> gatherer;
    (*iter)->parens->accept(&gatherer);
    for (auto iter2 = gatherer.gathered.begin(); iter2 != gatherer.gathered.end();
        ++iter2) {
      if (iter2 != gatherer.gathered.begin()) {
        middle(", ");
      }
      middle((*iter2)->arg);
    }
    finish("); } catch (std::bad_cast) {}");
  }
  line("throw std::bad_cast();");
  out();
  finish("};\n");
}
