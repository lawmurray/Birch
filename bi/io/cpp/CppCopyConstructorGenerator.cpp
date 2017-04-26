/**
 * @file
 */
#include "bi/io/cpp/CppCopyConstructorGenerator.hpp"

bi::CppCopyConstructorGenerator::CppCopyConstructorGenerator(
    std::ostream& base, const int level, const bool header) :
    CppBaseGenerator(base, level, header),
    before(false) {
  //
}

void bi::CppCopyConstructorGenerator::visit(const TypeParameter* o) {
  if (header) {
    line("template<class Frame = EmptyFrame>");
    start(o->name << "(const " << o->name << "<Group>& o");
    middle(", const bool deep = true");
    middle(", const Frame& frame = EmptyFrame()");
    middle(", const char* name = nullptr");
    middle(", const MemoryGroup& group = MemoryGroup())");

    if (!o->base->isEmpty()) {
      finish(" :");
      in();
      in();
      start("base_type(o, deep, frame, name, group)");
      before = true;
    }
    *this << o->braces;
    if (before) {
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
  if (before) {
    finish(',');
  } else {
    finish(": ");
    in();
    in();
  }
  start(o->param->name << "(o." << o->param->name);
  middle(", deep, frame, name, group)");
  before = true;
}

void bi::CppCopyConstructorGenerator::visit(const FuncDeclaration* o) {
  //
}
