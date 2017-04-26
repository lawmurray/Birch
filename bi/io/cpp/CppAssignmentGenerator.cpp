/**
 * @file
 */
#include "bi/io/cpp/CppAssignmentGenerator.hpp"

bi::CppAssignmentGenerator::CppAssignmentGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppAssignmentGenerator::visit(const TypeParameter* o) {
  /* basic assignment operator */
  if (!header) {
    line("template<class Group>");
    start("bi::type::");
  } else {
    start("");
  }
  middle(o->name << "<Group>& ");
  if (!header) {
    middle("bi::type::" << o->name << "<Group>::");
  }
  middle("operator=(const " << o->name << "<Group>& o_)");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();
    if (!o->base->isEmpty()) {
      line("base_type::operator=(o_);");
    }
    *this << o->braces;
    line("");
    line("return *this;");
    out();
    line("}\n");
  }

  /* generic assignment operator */
  if (header) {
    line("template<class Group1>");
    start(o->name << "<Group>&");
    middle(" operator=(const " << o->name << "<Group1>& o_)");
    finish(" {");
    in();
    if (!o->base->isEmpty()) {
      line("base_type::operator=(o_);");
    }
    *this << o->braces;
    line("");
    line("return *this;");
    out();
    line("}\n");
  }
}

void bi::CppAssignmentGenerator::visit(const FuncParameter* o) {
  //
}

void bi::CppAssignmentGenerator::visit(const VarParameter* o) {
  middle(o->name << " = o_." << o->name);
}
