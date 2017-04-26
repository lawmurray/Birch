/**
 * @file
 */
#include "bi/io/cpp/CppViewConstructorGenerator.hpp"

bi::CppViewConstructorGenerator::CppViewConstructorGenerator(
    std::ostream& base, const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppViewConstructorGenerator::visit(const TypeParameter* o) {
  if (header) {
    line("template<class Frame, class View>");
    start("");
    middle(o->name);
    middle("(const " << o->name << "<Group>& o");
    middle(", const Frame& frame");
    middle(", const View& view)");

    finish(" :");
    in();
    in();
    if (!o->base->isEmpty()) {
      line("base_type(o, frame, view),");
    }
    start("group(o.group)");
    *this << o->braces;
    out();
    out();
    finish(" {");
    in();
    line("//");
    out();
    line("}\n");
  }
}

void bi::CppViewConstructorGenerator::visit(const VarDeclaration* o) {
  finish(',');
  start(o->param->name << "(o." << o->param->name << ", frame, view)");
}

void bi::CppViewConstructorGenerator::visit(const FuncDeclaration* o) {
  //
}
