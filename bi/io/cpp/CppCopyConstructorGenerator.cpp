/**
 * @file
 */
#include "bi/io/cpp/CppCopyConstructorGenerator.hpp"

bi::CppCopyConstructorGenerator::CppCopyConstructorGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppCopyConstructorGenerator::visit(const Class* o) {
  if (!header) {
    genTemplateParams(o);
    start("bi::type::" << o->name);
    genTemplateArgs(o);
    middle("::");
  } else {
    start("");
  }
  middle(o->name);
  CppBaseGenerator aux(base, level, header);
  aux << "(const this_type& o, const world_t world)";
  if (header) {
    finish(";\n");
  } else {
    finish(" :");
    in();
    in();
    start("super_type(o, world)");
    for (auto iter = o->params->begin(); iter != o->params->end(); ++iter) {
      *this << *iter;
    }
    *this << o->braces->strip();
    out();
    out();
    finish(" {");
    in();
    line("//");
    out();
    line("}\n");
  }
}

void bi::CppCopyConstructorGenerator::visit(const MemberParameter* o) {
  finish(',');
  start(o->name << "(o." << o->name << ", world)");
}

void bi::CppCopyConstructorGenerator::visit(const MemberVariable* o) {
  finish(',');
  start(o->name << "(o." << o->name << ", world)");
}

void bi::CppCopyConstructorGenerator::visit(const MemberFunction* o) {
  //
}

void bi::CppCopyConstructorGenerator::visit(const MemberFiber* o) {
  //
}

void bi::CppCopyConstructorGenerator::visit(const ConversionOperator* o) {
  //
}

void bi::CppCopyConstructorGenerator::visit(const AssignmentOperator* o) {
  //
}
