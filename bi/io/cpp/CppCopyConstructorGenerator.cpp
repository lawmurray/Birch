/**
 * @file
 */
#include "bi/io/cpp/CppCopyConstructorGenerator.hpp"
#include "bi/io/cpp/CppBaseGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"

bi::CppCopyConstructorGenerator::CppCopyConstructorGenerator(
    std::ostream& base, const int level, const bool header) :
    indentable_ostream(base, level, header) {
  //
}

void bi::CppCopyConstructorGenerator::visit(const ModelParameter* o) {
  if (!header) {
    line("template<class Group>");
    start("bi::model::" << o->name->str() << "<Group>::");
  } else {
    start("");
  }
  middle(
      o->name->str() << "(const " << o->name->str() << "<Group>& o)");
  if (header) {
    finish(";\n");
  } else {
    if (!o->base->isEmpty() || o->vars().size() > 0) {
      finish(" :");
      in();
      in();
      if (!o->base->isEmpty()) {
        CppBaseGenerator aux(base, level, header);
        aux << o->base;
        finish("(o),");
      }
      start("group(o.group)");

      Gatherer<VarDeclaration> gatherer;
      o->braces->accept(&gatherer);
      for (auto iter = gatherer.gathered.begin(); iter != gatherer.gathered.end(); ++iter) {
        *this << *iter;
      }

      out();
      out();
    }
    finish(" {");
    in();
    line("//");
    out();
    line("}\n");
  }
}

void bi::CppCopyConstructorGenerator::visit(const VarDeclaration* o) {
  finish(',');
  start(o->param->name->str() << "(o." << o->param->name->str() << ")");
}
