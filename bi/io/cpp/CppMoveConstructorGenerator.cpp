/**
 * @file
 */
#include "bi/io/cpp/CppMoveConstructorGenerator.hpp"
#include "bi/io/cpp/CppBaseGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"

bi::CppMoveConstructorGenerator::CppMoveConstructorGenerator(
    std::ostream& base, const int level, const bool header) :
    indentable_ostream(base, level, header) {
  //
}

void bi::CppMoveConstructorGenerator::visit(const ModelParameter* o) {
  if (!header) {
    line("template<class Group>");
    start("bi::model::" << o->name->str() << "<Group>::");
  } else {
    start("");
  }
  middle(o->name->str() << "(" << o->name->str() << "<Group>&& o)");
  if (header) {
    finish(";\n");
  } else {
    Gatherer<VarDeclaration> gatherer;
    o->braces->accept(&gatherer);
    if (!o->base->isEmpty() || gatherer.gathered.size() > 0) {
      finish(" :");
      in();
      in();
      if (!o->base->isEmpty()) {
        finish("base_type(o),");
      }
      start("group(o.group)");
      for (auto iter = gatherer.gathered.begin();
          iter != gatherer.gathered.end(); ++iter) {
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

void bi::CppMoveConstructorGenerator::visit(const VarDeclaration* o) {
  finish(',');
  start(o->param->name->str() << "(o." << o->param->name->str() << ')');
}
