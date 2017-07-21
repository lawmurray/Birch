/**
 * @file
 */
#include "bi/io/cpp/CppCloneGenerator.hpp"

bi::CppCloneGenerator::CppCloneGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppCloneGenerator::visit(const Class* o) {
  if (!header) {
    start("bi::type::");
  } else {
    start("virtual ");
  }
  middle(o->name << "* ");
  if (!header) {
    middle("bi::type::" << o->name << "::");
  }
  middle("clone()");
  if (header) {
    finish(";\n");
  } else {
    finish(" {");
    in();
    line("return copy_object(this);");
    out();
    line("}\n");
  }
}

void bi::CppCloneGenerator::visit(const MemberParameter* o) {
  //
}

void bi::CppCloneGenerator::visit(const MemberVariable* o) {
  //
}

void bi::CppCloneGenerator::visit(const MemberFunction* o) {
  //
}

void bi::CppCloneGenerator::visit(const ConversionOperator* o) {
  //
}

void bi::CppCloneGenerator::visit(const AssignmentOperator* o) {
  //
}
