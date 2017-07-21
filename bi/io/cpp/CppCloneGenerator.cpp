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
    start("");
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
    line("auto result = copy_object(this);");
    /// @todo What if more than one member attribute points to the same
    /// object, or even this?
    *this << o->braces;
    line("return result;");
    out();
    line("}\n");
  }
}

void bi::CppCloneGenerator::visit(const MemberParameter* o) {
  if (o->type->isClass()) {
    line("result->" << o->name << "->use();");
  }
}

void bi::CppCloneGenerator::visit(const MemberVariable* o) {
  if (o->type->isClass()) {
    line("result->" << o->name << "->use();");
  }
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
